from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


class PowerGridDataset(Dataset):
    """
    Torch Dataset that owns the data pipeline for power-grid graph samples.

    Provides:
        - .data_list: list[torch_geometric.data.Data]
        - train/valid/test index splits
        - feature standardization stats computed from train split

    Each sample (Data) contains:
        x:         (nbus, 2)   [Pd, Qd] in per-unit
        y:         (nbus, 4)   [Pg@spv, Qg@spv, |V|, angle] stored per-bus (zeros where not applicable)
        node_mask: (nbus,)     True for spv buses (generator/slack buses)
        edge_index:(2, nl)     [from_bus; to_bus]
        edge_attr: (nl, 4)     [R, X, B, RateA_pu]
    """

    def __init__(
        self,
        mat_filename: str,
        ctx: "GridContext",
        *,
        split: Tuple[float, float, float] = (0.8, 0.2, 0.0),
        seed: int = 0,
        max_samples: Optional[int] = None,
        drop_nan: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.ctx = ctx
        self.dtype = dtype

        train_frac, valid_frac, test_frac = split
        if abs((train_frac + valid_frac + test_frac) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")

        data = spio.loadmat(mat_filename)
        dem = data["Dem"].T / ctx.baseMVA
        gen = data["Gen"].T / ctx.genbase
        vol = data["Vol"].T

        # Remove NaN rows early (raw arrays)
        if drop_nan:
            feas_mask = ~np.isnan(dem).any(axis=1)
            feas_mask &= ~np.isnan(gen).any(axis=1)
            feas_mask &= ~np.isnan(vol).any(axis=1)
            dem = dem[feas_mask]
            gen = gen[feas_mask]
            vol = vol[feas_mask]

        if max_samples is not None:
            dem = dem[:max_samples]
            gen = gen[:max_samples]
            vol = vol[:max_samples]

        self._data_list = self._build_graph_data(dem, gen, vol)

        # Final filter on constructed y (covers edge-cases)
        if drop_nan:
            self._data_list = [d for d in self._data_list if not torch.isnan(d.y).any()]

        self.num_samples = len(self._data_list)
        self.splits = self._make_splits(self.num_samples, train_frac, valid_frac, test_frac, seed)

        # Standardization stats (computed from split)
        self.node_mean, self.node_std, self.edge_mean, self.edge_std = self._compute_standardization(
            indices=self.splits.train
        )

    @property
    def data_list(self) -> List[Data]:
        return self._data_list

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]

    def get_split_indices(self) -> DatasetSplit:
        return self.splits

    def _build_graph_data(self, demand: np.ndarray, gen: np.ndarray, voltage: np.ndarray) -> List[Data]:
        n = demand.shape[0]
        nbus = self.ctx.nbus
        nl = self.ctx.nl

        spv = self.ctx.indices.spv
        nonslack = self.ctx.indices.nonslack_bus

        # Node features (Pd, Qd) in per-unit
        node_x = np.zeros((n, nbus, 2), dtype=np.float64)
        node_mask = np.zeros((n, nbus), dtype=np.bool_)

        for s in range(n):
            node_x[s, :, 0] = np.real(demand[s, :])
            node_x[s, :, 1] = np.imag(demand[s, :])
            node_mask[s, spv] = True

        # Edge features: [from, to, R, X, B, RateA_pu]
        edge_feat = np.zeros((n, nl, 6), dtype=np.float64)
        f_bus = self.ctx.ppc["branch"][:, 0].astype(np.int64, copy=False)
        t_bus = self.ctx.ppc["branch"][:, 1].astype(np.int64, copy=False)

        br = self.ctx.ppc["branch"]
        edge_feat[:, :, 0] = f_bus
        edge_feat[:, :, 1] = t_bus
        edge_feat[:, :, 2] = br[:, 2]  # BR_R
        edge_feat[:, :, 3] = br[:, 3]  # BR_X
        edge_feat[:, :, 4] = br[:, 4]  # BR_B

        rateA = (br[:, 5] / self.ctx.baseMVA).astype(np.float64, copy=False)  # RATE_A/baseMVA
        edge_feat[:, :, 5] = rateA

        # Targets y per-bus: [Pg, Qg, |V|, theta]
        # Pg/Qg are only meaningful at spv buses; stored per-bus with zeros elsewhere.
        node_y = np.zeros((n, nbus, 4), dtype=np.float64)
        for s in range(n):
            node_y[s, spv, 0] = np.real(gen[s, :])     # Pg on spv
            node_y[s, spv, 1] = np.imag(gen[s, :])     # Qg on spv
            node_y[s, :, 2] = np.abs(voltage[s, :])    # Vm on all buses
            node_y[s, :, 3] = np.angle(voltage[s, :])  # Va on all buses

        # Build Data objects
        out: List[Data] = []
        for s in range(n):
            x_t = torch.tensor(node_x[s], dtype=self.dtype)
            y_t = torch.tensor(node_y[s], dtype=self.dtype)
            mask_t = torch.tensor(node_mask[s], dtype=torch.bool)

            edge_index = torch.tensor(edge_feat[s, :, 0:2].T, dtype=torch.long)
            edge_attr = torch.tensor(edge_feat[s, :, 2:], dtype=self.dtype)  # [R,X,B,RateA_pu]

            out.append(
                Data(
                    x=x_t,
                    y=y_t,
                    node_mask=mask_t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
            )
        return out

    def _make_splits(
        self,
        n: int,
        train_frac: float,
        valid_frac: float,
        test_frac: float,
        seed: int,
    ) -> DatasetSplit:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        n_train = int(round(n * train_frac))
        n_valid = int(round(n * valid_frac))
        n_test = n - n_train - n_valid

        train_idx = perm[:n_train]
        valid_idx = perm[n_train:n_train + n_valid]
        test_idx = perm[n_train + n_valid:n_train + n_valid + n_test]

        return DatasetSplit(train=train_idx, valid=valid_idx, test=test_idx)

    def _compute_standardization(self, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stack nodes/edges across selected samples
        nbus = self.ctx.nbus
        nl = self.ctx.nl

        node_stack = torch.zeros((len(indices) * nbus, 2), dtype=self.dtype)
        edge_stack = torch.zeros((len(indices) * nl, 4), dtype=self.dtype)

        for i, idx in enumerate(indices):
            d = self._data_list[int(idx)]
            node_stack[i * nbus:(i + 1) * nbus, :] = d.x
            edge_stack[i * nl:(i + 1) * nl, :] = d.edge_attr

        node_mean = node_stack.mean(dim=0, keepdim=True)
        node_std = node_stack.std(dim=0, keepdim=True).clamp_min(1e-12)
        edge_mean = edge_stack.mean(dim=0, keepdim=True)
        edge_std = edge_stack.std(dim=0, keepdim=True).clamp_min(1e-12)

        return node_mean, node_std, edge_mean, edge_std

    def standardize(self, data: Data) -> Data:
        """
        Return a standardized copy of `data` using train-split statistics.
        """
        x = (data.x - self.node_mean) / self.node_std
        e = (data.edge_attr - self.edge_mean) / self.edge_std

        out = Data(
            x=x,
            y=data.y,
            node_mask=data.node_mask,
            edge_index=data.edge_index,
            edge_attr=e,
        )
        return out
