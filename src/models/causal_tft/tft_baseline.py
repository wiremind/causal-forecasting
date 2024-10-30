import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.causal_tft.tft_core import TFTBackbone
from src.models.causal_tft.utils import create_sequential_layers


class TFTBaseline(TFTBackbone, pl.LightningModule):
    def __init__(
        self,
        projection_length: int,
        last_nn: list[int],
        horizon: int,
        static_features_size: int,
        temporal_features_size: int,
        target_size: int = 1,
        hidden_size: int = 128,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
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
            projection_length (int): Number of time steps to forecast at once
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
        super().__init__(
            horizon=horizon,
            static_features_size=static_features_size,
            temporal_features_size=temporal_features_size,
            target_size=target_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            static_embedding_sizes=static_embedding_sizes,
            temporal_embedding_sizes=temporal_embedding_sizes,
            trend_size=trend_size,
            n_static_layers=n_static_layers,
            n_att_layers=n_att_layers,
            conv_padding_size=conv_padding_size,
            conv_blocks=conv_blocks,
            kernel_size=kernel_size,
        )

        self.regression_loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.projection_length = projection_length

        # Creates the final dense network
        self.last_nn = create_sequential_layers(last_nn, self.hidden_size, target_size)

        self.configure_optimizers()
        self.save_hyperparameters()

    def format_batch_window(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        """
        Format the window that should be sent to a TFT model
        The window is composed of:
        - insample_y: The temporal features known only in the past
        - multivariate_exog: Temporal features known both in the past and in the futur
        - stat_exog: static features
        """
        vitals = batch["vitals"].float()
        static_features = batch["static_features"].float()
        treatments = batch["current_treatments"].float()
        y = batch["outputs"].float()
        active_entries = torch.zeros_like(y)
        li_insample_y = batch["outputs"].clone().float()

        # Number of hours passed in emergency
        position = torch.arange(batch["vitals"].shape[1])
        position = position.repeat(batch["vitals"].shape[0], 1, 1)
        position = torch.permute(position, (0, 2, 1)).to(batch["vitals"].device)

        taus = batch["future_past_split"].int()
        for i, tau in enumerate(taus):
            li_insample_y[i, tau:] = 0
            active_entries[i, tau : tau + self.projection_length] = 1
            vitals[i, tau:] = 0

        temporal = torch.concat([vitals, position, treatments], dim=-1)
        # Encapsulating inputs
        windows = {}
        windows["insample_y"] = li_insample_y
        windows["multivariate_exog"] = temporal
        windows["stat_exog"] = static_features

        return windows, taus, y, active_entries, treatments

    def forward(self, windows_batch: dict[str, Tensor], tau: Tensor | None = None):
        z = self.get_z(windows_batch, tau)
        y_hat = self.last_nn(z)
        return y_hat

    def forecast(self, batch: dict[str, Tensor]):
        with torch.no_grad():
            windows, taus, _, _, _ = self.format_batch_window(batch)
            output = self.forward(windows, taus)
        return output

    def loss(self, y: Tensor, y_hat: Tensor, active_entries: Tensor) -> Tensor:
        loss = F.mse_loss(y_hat, y, reduction="none")
        loss = (active_entries * loss).sum() / active_entries.sum()
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        self.train()
        windows, taus, y, active_entries, _ = self.format_batch_window(batch)
        output = self.forward(windows, taus.int())
        loss = self.loss(output, y, active_entries)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        self.eval()
        with torch.no_grad():
            windows, taus, y, active_entries, _ = self.format_batch_window(batch)
            output = self.forward(windows, taus)
            loss = self.loss(output, y, active_entries)
            self.log(f"val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # Required by torch lightning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer
