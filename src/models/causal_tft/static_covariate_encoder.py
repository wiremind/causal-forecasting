from torch import Tensor, nn
from torch.nn import functional as F

from src.models.causal_tft.utils import VariableSelectionNetwork


class StaticCovariateEncoder(nn.Module):
    def __init__(
        self, hidden_size: int, num_static_vars: int, dropout: float, n_layers: int
    ):
        """This class is the encoder of the static features

        Args:
            hidden_size (int): hidden size of the mode
            num_static_vars (int): Number of static features in the model
            dropout (float): dropout in the hidden layers encoding static features
            n_layers (int): number of layers for encoding static features
        """
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            hidden_size=hidden_size,
            num_inputs=num_static_vars,
            dropout=dropout,
            n_layers=n_layers,
        )
        self.context_grns = GRN(
            input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
        )
        self.drop_rate = dropout

    def forward(self, x: Tensor):
        variable_ctx = self.vsn(x)
        context_static = self.context_grns(variable_ctx)
        return context_static


class GLU(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size: int | None, hidden_size: int, eps: float):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = nn.LayerNorm(output_size if output_size else hidden_size, eps=eps)

    def forward(self, x):
        return self.ln(x)


class GRN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_hidden_size: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Tensor | None = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if not self.out_proj else self.out_proj(a)
        x = x + y
        x = self.layer_norm(x)
        return x
