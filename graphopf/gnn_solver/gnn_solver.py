from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import ChebConv
from torch_geometric.utils import to_dense_batch

from gnn_solver.grid_constants import GridConstants
from gnn_solver.edge_aggregation import EdgeAggregation


@dataclass(frozen=True)
class GNNSolverConfig:
    input_dim: int
    edge_feature_dim: int
    hidden_dim: int
    n_gnn_layers: int
    K: int
    dropout_rate: float
    output_dim: int


class GNNSolver(nn.Module):
    def __init__(
        self,
        grid: GridConstants,
        config: GNNSolverConfig,
    ) -> None:
        super().__init__()

        self.grid = grid
        self.config = config

        self.nbus = grid.nbus
        self.npv = grid.indices.npv
        self.nslack = grid.indices.nslack
        self.spv_size = self.npv + self.nslack

        self.layers = nn.ModuleList()

        self.layers.append(
            EdgeAggregation(
                nfeature_dim=config.input_dim,
                efeature_dim=config.edge_feature_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
            )
        )
        self.layers.append(ChebConv(config.hidden_dim, config.hidden_dim, K=config.K, bias=True))

        for _ in range(config.n_gnn_layers - 1):
            self.layers.append(
                EdgeAggregation(
                    nfeature_dim=config.hidden_dim,
                    efeature_dim=config.edge_feature_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.hidden_dim,
                )
            )
            self.layers.append(ChebConv(config.hidden_dim, config.hidden_dim, K=config.K, bias=True))

        self.flatten = nn.Linear(self.spv_size * config.hidden_dim, config.output_dim, bias=True)
        self.dropout = nn.Dropout(config.dropout_rate, inplace=False)


    def forward(self, data: Batch) -> torch.Tensor:
        num_graphs = int(data.num_graphs)

        # data.x: (total_nodes, input_dim). For ACOPF loader input_dim is typically 2: [Pd, Qd]
        x_dense, _ = to_dense_batch(data.x, batch=data.batch)  # (B, max_nodes, input_dim)

        # Physics input = [Pd | Qd]
        inputs = torch.cat([x_dense[:, :, 0], x_dense[:, :, 1]], dim=1)  # (B, 2*nbus)

        x = data.x
        edge_index = data.edge_index
        e_attr = data.edge_attr

        for i in range(0, len(self.layers), 2):
            edge_layer = self.layers[i]
            cheb_layer = self.layers[i + 1]

            x = edge_layer(x=x, edge_index=edge_index, edge_attr=e_attr)
            x = torch.relu(x)
            x = self.dropout(x)

            x = cheb_layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x_spv = x[data.node_mask, :]  # (B*spv_size, hidden_dim)
        x_flat = x_spv.view(num_graphs, self.spv_size * self.config.hidden_dim)

        guess = self.flatten(x_flat)
        solution = self.physics_layer(inputs, guess)
        return solution

