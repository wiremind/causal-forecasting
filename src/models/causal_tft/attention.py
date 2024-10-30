import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, example_length, attn_dropout, dropout):
        super().__init__()
        self.n_heads = n_heads
        assert hidden_size % n_heads == 0
        self.d_head = hidden_size // n_heads
        self.hidden_size = hidden_size
        self.qkv_linears = nn.Linear(
            hidden_size, (2 * self.n_heads + 1) * self.d_head, bias=False
        )
        self.q = nn.Linear(hidden_size, self.n_heads * self.d_head)
        self.k = nn.Linear(hidden_size, self.n_heads * self.d_head)
        self.v = nn.Linear(hidden_size, self.d_head)
        self.out_proj = nn.Linear(self.d_head, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5
        self.register_buffer(
            "_mask",
            torch.triu(
                torch.full((example_length, example_length), float("-inf")), 1
            ).unsqueeze(0),
        )

    def forward(
        self,
        x: Tensor,
        mask_future_timesteps: bool = True,
        return_weights: bool = False,
        static_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # [Batch,Time,MultiHead,AttDim] := [N,T,M,AD]
        # Computation of the queries, keys and values
        context = static_features if static_features is not None else x
        bs, t, _ = x.shape
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)

        q = q.view(bs, t, self.n_heads, self.d_head)
        k = k.view(bs, t, self.n_heads, self.d_head)
        v = v.view(bs, t, self.d_head)

        # [N,T1,M,Ad] x [N,T2,M,Ad] -> [N,M,T1,T2]
        # attn_score = torch.einsum('bind,bjnd->bnij', q, k)
        # Computation of the context vectores
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)

        # Masking the future
        if mask_future_timesteps:
            attn_score = attn_score + self._mask
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)

        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)
        if return_weights == True:
            return out, attn_vec, attn_prob
        return out, attn_vec


class BasicAttentionBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        example_length: int,
        attn_dropout: float,
        dropout: float,
        static_size: int | None = None,
    ) -> None:
        super().__init__()
        if static_size is None:
            static_size = hidden_size
        self.attention1 = InterpretableMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            example_length=example_length,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )
        self.static_encoder = nn.Linear(static_size, hidden_size)
        self.attention2 = InterpretableMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            example_length=example_length,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor, static_features: Tensor | None = None) -> Tensor:
        # attention standard
        N, L, _ = x.shape
        x = x + self.attention1(self.norm1(x), mask_future_timesteps=True)[0]
        # Conditional attention
        if not (static_features is None):
            N, H = static_features.shape
            static_features = static_features.unsqueeze(1).expand(N, L, H)
            x = (
                x
                + self.attention2(
                    self.norm2(x),
                    mask_future_timesteps=True,
                    static_features=F.relu(self.static_encoder(static_features)),
                )[0]
            )
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        example_length: int,
        attn_dropout: float,
        dropout: float,
        static_size: int | None = None,
        n_layers: int = 4,
    ):
        """This class is one attention block

        Args:
            n_heads (int): Number of heads of attentions
            hidden_size (int): Hidden size of the model
            example_length (int): Length of the sequence
            attn_dropout (float): Dropout in the layer attention
            dropout (float): Dropout in other layers
            n_layers (int, optional): Number of attention layers. Defaults to 4.
        """
        super(AttentionBlock, self).__init__()
        li = []
        for _ in range(n_layers):
            li.append(
                BasicAttentionBlock(
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    example_length=example_length,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    static_size=static_size,
                )
            )
            li.append(nn.LayerNorm(normalized_shape=hidden_size))
        self.li_attention = nn.ModuleList(li)

    def forward(self, x: Tensor, static_features: Tensor | None = None) -> Tensor:
        for i in range(len(self.li_attention) // 2):
            out = self.li_attention[2 * i](x, static_features)
            x = x + self.li_attention[2 * i + 1](out)
        return x
