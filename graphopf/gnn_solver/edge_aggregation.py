import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import aggr
from torch_geometric.utils import degree

class EdgeAggregation(MessagePassing):
    def __init__(self, nfeature_dim: int, efeature_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            aggr=aggr.MultiAggregation(
                aggrs=["mean", "std"],
                mode="proj",
                mode_kwargs=dict(in_channels=hidden_dim, out_channels=hidden_dim),
            )
        )

        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim * 2 + efeature_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_aggr(
            torch.cat(
                [
                    norm.view(-1, 1) * x_i,
                    norm.view(-1, 1) * x_j,
                    norm.view(-1, 1) * edge_attr,
                ],
                dim=-1,
            )
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)