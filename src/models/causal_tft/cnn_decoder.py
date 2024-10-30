from encodec.modules.conv import SConvTranspose1d
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.causal_tft.attention import AttentionBlock
from src.models.causal_tft.utils import Resblock


class CNNDecoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        horizon: int,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        n_att_layers: int = 1,
        padding_length: int = 128,
        n_blocks: int = 2,
        kernel_size: int = 7,
    ):
        """
        This class gathers the transformer bottleneck of the model and the CNN encoder

        Args:
            n_heads (int): number of heads in the attention mechanism
            hidden_size (int): size of the hidden layers
            horizon (int): size of the horizon to forecast
            attn_dropout (float): dropout rate in the attention mechanism
            dropout (float): dropout rate in the network layers
            n_att_layers (int): number of layers in the attention mechanism
            padding_length (int): length of the input channels sent to the convolution. If the horizon
            of the time series is lower than the padding_length we use zero padding to adapt the size
            n_blocks (int): number of EncoderBlock
            kernel_size (int): size of the kernel in the convolution

        """
        super().__init__()
        self.horizon = horizon

        # Bottleneck attention
        self.attention = AttentionBlock(
            n_heads=n_heads,
            hidden_size=hidden_size * (n_blocks + 1),
            example_length=padding_length // (2**n_blocks),
            attn_dropout=attn_dropout,
            dropout=dropout,
            n_layers=n_att_layers,
            static_size=hidden_size,
        )

        # Upsampling blocks
        input_channels = hidden_size * (n_blocks + 1)
        padding_length = padding_length // (2**n_blocks) * 2
        li_up = []
        for _ in range(n_blocks):
            li_up.append(
                DecoderBlock(
                    n_heads=n_heads,
                    input_channels=input_channels,
                    output_channels=input_channels - hidden_size,
                    origin_size=hidden_size,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    n_att_layers=n_att_layers,
                    padding_length=padding_length,
                    kernel_size=kernel_size,
                )
            )
            input_channels -= hidden_size
            padding_length = padding_length * 2
        self.li_up = nn.ModuleList(li_up)

    def forward(self, temporal_features: Tensor, static_features: Tensor):
        # Temporal self attention
        enriched = temporal_features[-1]
        x = self.attention(enriched)
        for i in range(1, len(self.li_up) + 1):
            x = self.li_up[i - 1](x, temporal_features[-i - 1], static_features)

        # We only keep the h first tokens because we did padding in the encoder
        x = x[:, : self.horizon]
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        input_channels: int,
        output_channels: int,
        origin_size: int,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        n_att_layers: int = 1,
        padding_length: int = 128,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.up = Upsample(input_channels, output_channels, dropout, kernel_size)
        self.norm1 = nn.LayerNorm(output_channels)
        self.resblock1 = Resblock(output_channels, dropout, kernel_size)
        self.resblock2 = Resblock(output_channels, dropout, kernel_size)
        self.attention = AttentionBlock(
            n_heads,
            output_channels,
            padding_length,
            attn_dropout,
            dropout,
            n_layers=n_att_layers,
        )
        self.norm2 = nn.LayerNorm(output_channels)
        self.context_adanorm = nn.Sequential(
            nn.Linear(origin_size, output_channels * 2), nn.GELU()
        )

    def forward(self, x, skip_connection, static_features=None):
        x = self.up(x)
        x = self.resblock1(x)
        if not (static_features is None):
            N, _, h = x.shape
            weights_stat = self.context_adanorm(static_features).reshape(N, 1, h, 2)
            x = weights_stat[:, :, :, 0] * self.norm1(x) + weights_stat[:, :, :, 1]
        x = self.norm2(x + skip_connection)
        x = self.resblock2(x)
        x = self.attention(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float = 0.1,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.cnn = SConvTranspose1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            causal=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = F.gelu(self.cnn(y))
        y = y.transpose(1, 2)
        y = self.dropout(self.norm(y))

        return y
