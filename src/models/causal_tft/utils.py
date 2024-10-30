import copy

import torch
import torch.nn.functional as F
from encodec.modules.conv import SConv1d
from torch import Tensor, nn


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self, hidden_size: int, num_inputs: int, dropout: float, n_layers: int = 1
    ):
        super().__init__()
        seq = [nn.Linear(num_inputs * hidden_size, 4 * hidden_size), nn.GELU()]
        for _ in range(n_layers):
            seq.extend(
                [
                    nn.Linear(4 * hidden_size, 4 * hidden_size),
                    nn.Dropout(dropout),
                    nn.GELU(),
                ]
            )
        seq.append(nn.Linear(4 * hidden_size, hidden_size))
        self.seq = nn.Sequential(*seq)
        self.lin_c = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, context=None):
        x = x.flatten(start_dim=-2)
        x = self.seq(x)
        if context is not None:
            x = torch.cat([self.lin_c(context).unsqueeze(1), x], dim=-2)
        return x


class Resblock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.cnn1 = SConv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            causal=True,
            stride=1,
            pad_mode="constant",
        )
        self.dropout = nn.Dropout1d(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.cnn2 = SConv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            causal=True,
            stride=1,
            pad_mode="constant",
        )

    def forward(self, x: Tensor) -> Tensor:
        y = x.transpose(1, 2)
        y = F.gelu(self.cnn1(y))
        y = F.gelu(self.cnn2(y))
        y = self.dropout(self.norm(y.transpose(1, 2)))
        y = x + y
        return y


def create_sequential_layers(nn_final_m, hidden_size: int, target_size: int):
    nn_final = copy.deepcopy(nn_final_m)
    nn_final.append(target_size)
    last_nn = nn.Sequential()
    last_nn.append(nn.Linear(in_features=hidden_size, out_features=nn_final[0]))
    for i in range(len(nn_final) - 1):
        last_nn.append(nn.ReLU())
        last_nn.append(nn.Linear(in_features=nn_final[i], out_features=nn_final[i + 1]))

    return last_nn
