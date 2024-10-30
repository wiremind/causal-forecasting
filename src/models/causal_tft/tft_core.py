import torch
from torch import Tensor, nn

from src.models.causal_tft.cnn_decoder import CNNDecoder
from src.models.causal_tft.cnn_encoder import CausalCNNEncoder
from src.models.causal_tft.embeding import TFTEmbedding
from src.models.causal_tft.static_covariate_encoder import \
    StaticCovariateEncoder


class TFTBackbone(nn.Module):
    def __init__(
        self,
        horizon: int,
        static_features_size: int,
        temporal_features_size: int,
        target_size: int = 1,
        hidden_size: int = 128,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        static_embedding_sizes: list | None = None,
        temporal_embedding_sizes: list | None = None,
        trend_size: int = 1,  # TODO rename in something like past_features_size
        n_static_layers: int = 2,
        n_att_layers: int = 4,
        conv_padding_size: int = 128,
        conv_blocks: int = 2,
        kernel_size: int = 7,
    ):
        """This class is an abstract class. It contains the key components of a TFT. It is also a pytorch lightning module, this
        framework of pytorch used to quickly train the models. This class is used as a parent of every model we work on.

        Args:
            horizon (int): Horizon of the prediction, how many days we want to predict at most.
            static_features_size (int): Number of features in the static metadata. This number should include the legnth of temporal_embedding_sizes
            temporal_features_size (int): Number of temporal features. This number should include the length of temporal_embedding_sizes. This number does not include
            the trend size
            Temporal features are used both before and after the prediction time step
            target_size (int): Number of head at the end of the network. Defaults to 1.
            hidden_size (int): Size of the hidden state of the network. Defaults to 128.
            n_heads (int): Number of head for the attention mechanism. Defaults to 4.
            attn_dropout (float): Dropout rate for the attention mechanism. Defaults to 0.0.
            dropout (float): Dropout rate. Defaults to 0.1.
            learning_rate (float): initial lr. Defaults to 1e-3.
            static_embedding_sizes (list, optional): List of the maximum values for the categorical features in the static features. Defaults to None.
            temporal_embedding_sizes (list, optional): List of the maximum values for the categorical features in the future data. Defaults to None.
            trend_size (int): size of the trend (number of dimensions/channels). Defaults is 2
            n_static_layers (int): number of layers used to embed static features. Defaults to 2
            n_att_layers (int): number of attention layers

        """
        super().__init__()
        self.hidden_size = hidden_size
        self.static_features_size = static_features_size
        self.temporal_features_size = temporal_features_size
        self.horizon = horizon
        self.target_size = target_size
        self.trend_size = trend_size

        self.static_embedding_sizes = static_embedding_sizes
        self.temporal_embedding_sizes = temporal_embedding_sizes

        self.embedding = TFTEmbedding(
            hidden_size=self.hidden_size,
            static_features_size=self.static_features_size,
            temporal_features_size=self.temporal_features_size,
            target_size=self.trend_size,
            static_embedding_sizes=self.static_embedding_sizes,
            temporal_embedding_sizes=self.temporal_embedding_sizes,
        )
        self.static_encoder = StaticCovariateEncoder(
            hidden_size=self.hidden_size,
            num_static_vars=self.static_features_size,
            dropout=dropout,
            n_layers=n_static_layers,
        )

        self.temporal_encoder = CausalCNNEncoder(
            hidden_size=self.hidden_size,
            temporal_features_size=(self.temporal_features_size + self.trend_size),
            n_heads=n_heads,
            dropout=dropout,
            padding_length=conv_padding_size,
            horizon=self.horizon,
            trend_size=self.trend_size,
            n_blocks=conv_blocks,
            n_att_layers=n_att_layers,
            attn_dropout=attn_dropout,
            kernel_size=kernel_size,
        )

        # ------------------------------ Decoders -----------------------------#
        self.temporal_fusion_decoder = CNNDecoder(
            n_heads=n_heads,
            hidden_size=self.hidden_size,
            horizon=self.horizon,
            attn_dropout=attn_dropout,
            dropout=dropout,
            n_att_layers=n_att_layers,
            padding_length=conv_padding_size,
            n_blocks=conv_blocks,
            kernel_size=kernel_size,
        )

    def get_z(self, windows_batch: dict[str, Tensor], tau: Tensor | None = None):
        """Function returning the latent variable of the batch.

        Args:
            window_batch (dict): contains all the data of the batch

        Returns:
            tensor: latent variable, output of the TFT
        """
        y_insample = windows_batch["insample_y"]  # trend timeseries
        multivariate = windows_batch["multivariate_exog"]  # futur time series
        stat_exog = windows_batch["stat_exog"]  # static features
        if tau is None:
            tau = torch.argmin(y_insample, dim=1)

        # Inputs embeddings
        s_inp, k_inp, t_observed_tgt = self.embedding(
            target_inp=y_insample,
            multi_exog=multivariate,
            stat_exog=stat_exog,
        )

        # -------------------------------- Inputs ------------------------------#
        # Static context
        static_features = self.static_encoder(s_inp)
        # We add the time serie to the temporale features
        temporal_features = torch.cat([k_inp, t_observed_tgt], dim=-2)

        # ---------------------------- Encode/Decode ---------------------------#
        # CNN
        temporal_features = self.temporal_encoder(
            temporal_features=temporal_features,
            static_features=static_features,
            tau=tau,
        )

        # Self-Attention decoder
        z = self.temporal_fusion_decoder(
            temporal_features=temporal_features, static_features=static_features
        )

        return z

    def forward(self, windows_batch: dict[str, Tensor], tau: Tensor | None = None):
        return self.get_z(windows_batch, tau)

    def predict(self, shaped_batch: tuple[Tensor, Tensor, Tensor]):
        li_insample_y, multivariate_exo, li_stat_exo = shaped_batch
        windows = {}
        windows["insample_y"] = li_insample_y
        windows["stat_exog"] = li_stat_exo
        windows["multivariate_exog"] = multivariate_exo

        return self.forward(windows)
