from encodec.modules.conv import SConv1d
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.causal_tft.attention import AttentionBlock
from src.models.causal_tft.utils import Resblock, VariableSelectionNetwork


class CausalCNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        temporal_features_size: int,
        horizon: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        padding_length: int = 0,
        trend_size: int = 1,
        n_blocks: int = 2,
        n_att_layers: int = 1,
        attn_dropout: float = 0.05,
        kernel_size: int = 7,
    ):
        """
        Causal CNN encoder to encode temporal data

        Args:
            hidden_size (int): size of the hidden layers
            temporal_features_size (int): Number of temporal features
            horizon (int): size of the horizon to forecast
            n_heads (int): number of heads in the attention mechanism
            dropout (float): dropout rate in the network layers
            padding_length (int): length of the input channels sent to the convolution. If the horizon
            of the time series is lower than the padding_length we use zero padding to adapt the size
            TODO: find a better name for padding parameter
            trend_size (int): TODO: complete the doc
            n_blocks (int): number of EncoderBlock
            n_att_layers (int): number of layers in the attention mechanism
            attn_dropout (float): dropout rate in the attention mechanism
            kernel_size (int): size of the kernel in the convolution

        """
        super().__init__()

        assert padding_length >= horizon

        self.trend_size = trend_size
        self.padding = nn.ZeroPad1d((0, padding_length - horizon))

        # Features selection for temporal features known everywhere
        self.temporal_vsn = VariableSelectionNetwork(
            hidden_size=hidden_size,
            num_inputs=temporal_features_size - self.trend_size,
            dropout=dropout,
        )
        # Features selection for temporal features known until tau
        self.past_vsn = VariableSelectionNetwork(
            hidden_size=hidden_size,
            num_inputs=temporal_features_size,
            dropout=dropout,
        )

        in_ch = hidden_size
        self.li_down = []
        for _ in range(n_blocks):
            self.li_down.append(
                EncoderBlock(
                    n_heads=n_heads,
                    in_channels=in_ch,
                    out_channels=in_ch + hidden_size,
                    origin_size=hidden_size,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    n_att_layers=n_att_layers,
                    padding=padding_length,
                    kernel_size=kernel_size,
                )
            )
            in_ch += hidden_size
            padding_length = padding_length // 2
        self.li_down = nn.ModuleList(self.li_down)

    def forward(
        self,
        temporal_features: Tensor,
        static_features: Tensor,
        tau: Tensor | None = None,
    ) -> list[Tensor]:
        """

        Args:
            temporal_features (Tensor): Temporal data, shape (batch_size, temporal length,
            n_features_temporal, hidden_size)
            static_features (Tensor): Static data, added to the sequence after the convolution, shape
            (batch_size, hidden_size)
            tau (Tensor, optional): Time step up until we know the time series. Defaults to None. Argument
            should be one dimensional int
        """
        # We apply a first features selection to the features known for the whole time series
        x = self.temporal_vsn(temporal_features[:, :, : -self.trend_size])
        # We then add the effect of the features known until tau
        if tau is not None:
            past_effect = self.past_vsn(temporal_features)
            for i in range(tau.shape[0]):
                x[i, : tau[i]] = x[i, : tau[i]] + past_effect[i, : tau[i]]

        # We apply the convolutions
        x = self.padding(x.transpose(1, 2)).transpose(1, 2)
        li_intermediate = [x]
        for i in range(len(self.li_down)):
            x = self.li_down[i](x, static_features)
            if i != len(self.li_down) - 1:
                li_intermediate.append(x)
        y = x
        li_intermediate.append(y)
        return li_intermediate


class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        in_channels: int,
        out_channels: int,
        origin_size: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.1,
        n_att_layers: int = 1,
        padding: int = 128,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.context_adanorm = nn.Sequential(
            nn.Linear(origin_size, in_channels * 2), nn.GELU()
        )
        self.down = Downsample(in_channels, out_channels, dropout, kernel_size)
        self.norm = nn.LayerNorm(in_channels)
        self.attention = AttentionBlock(
            n_heads=n_heads,
            hidden_size=in_channels,
            example_length=padding,
            attn_dropout=attn_dropout,
            dropout=dropout,
            static_size=origin_size,
            n_layers=n_att_layers,
        )
        self.resblock1 = Resblock(in_channels, dropout, kernel_size)
        self.resblock2 = Resblock(in_channels, dropout, kernel_size)

    def forward(self, x: Tensor, static_features: Tensor | None = None) -> Tensor:
        x = self.resblock1(x)
        x = self.attention(x, static_features)
        if not (static_features is None):
            N, _, h = x.shape
            weights_stat = self.context_adanorm(static_features).reshape(N, 1, h, 2)
            x = weights_stat[:, :, :, 0] * self.norm(x) + weights_stat[:, :, :, 1]
        x = self.resblock2(x)
        x = self.down(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.cnn1 = SConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            causal=True,
            stride=1,
            pad_mode="constant",
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = F.gelu(self.cnn1(x.transpose(1, 2))).transpose(1, 2)
        y = self.norm(y)
        y = self.dropout(y)
        # TODO: add a parameter for this
        return y[:, ::2, :]  ##Taking one token out of two
