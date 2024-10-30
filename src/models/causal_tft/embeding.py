import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TFTEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        static_features_size: int,
        temporal_features_size: int,
        static_embedding_sizes: list[int] | None,
        temporal_embedding_sizes: list[int] | None,
        target_size: int = 1,
    ):
        """
        This module is used to create the embeddings of the inputs passed to the TFT model
        The TFT model needs embedding of:
        - The static features
        - The temporal features
        - The target's trend

        Args:
            hidden_size (int): hidden size in embedding layers
            static_features_size (int): number of static features
            temporal_features_size (int): number of temporal features
            target_size (int): target size for the trend. Defaults is 1
            static_embedding_sizes (list[int] | None): list of unique values in categoircal features
            for the static features
            temporal_embedding_sizes (list[int] | None): list of unique values in categoircal features
            for the futur temporal features
        """
        super().__init__()
        # There are 4 types of input:
        # 1. Static continuous
        # 2. Temporal known a priori continuous
        # 3. Temporal observed continuous
        # 4. Temporal observed targets (time series obseved so far)

        self.hidden_size = hidden_size
        self.static_features_size = static_features_size
        self.target_size = target_size
        self.temporal_features_size = temporal_features_size

        static_embedding_sizes = static_embedding_sizes or []
        temporal_embedding_sizes = temporal_embedding_sizes or []

        self.n_static_categorical_features = len(static_embedding_sizes)
        self.n_temporal_categorical_features = len(temporal_embedding_sizes)

        # Design of the Embeddings for the static metadata
        self.static_embedding_sizes = static_embedding_sizes
        self.embedding_list_stat = nn.ModuleList(
            [nn.Embedding(size, hidden_size) for size in self.static_embedding_sizes]
        )
        # Design of the Embeddings for the temporal data
        self.temporal_embedding_sizes = temporal_embedding_sizes
        self.embedding_list_future = nn.ModuleList(
            [nn.Embedding(size, hidden_size) for size in self.temporal_embedding_sizes]
        )

        for attr, size in [
            ("stat_exog_embedding", static_features_size),
            ("multi_exog_embedding", temporal_features_size),
            ("tgt_embedding", target_size),
        ]:
            if size:
                vectors = nn.Parameter(Tensor(size, hidden_size))
                bias = nn.Parameter(torch.zeros(size, hidden_size))
                nn.init.xavier_normal_(vectors)
                setattr(self, attr + "_vectors", vectors)
                setattr(self, attr + "_bias", bias)
            else:
                setattr(self, attr + "_vectors", None)
                setattr(self, attr + "_bias", None)

    def _apply_embedding(
        self,
        cont: Tensor | None,
        cont_emb: Tensor,
        cont_bias: Tensor,
        is_stat_exog: bool = False,
        is_multivariate: bool = False,
    ):

        # Dimension augmentation for static data
        if cont is not None and is_stat_exog:
            # Continuous process
            continuous = cont[:, self.n_static_categorical_features :]
            cont_emb_continuous = cont_emb[self.n_static_categorical_features :]
            cont_bias_continuous = cont_bias[self.n_static_categorical_features :]
            continuous_transformed = (
                torch.mul(continuous.unsqueeze(-1), cont_emb_continuous)
                + cont_bias_continuous
            )

            # Categorical process
            categorical = cont[:, : self.n_static_categorical_features]
            embedding_representation = []
            if self.n_static_categorical_features > 0:
                for i in range(self.n_static_categorical_features):
                    embedding_representation.append(
                        self.embedding_list_stat[i](categorical[:, i])
                    )
                    embedding = F.gelu(torch.stack(embedding_representation, dim=1))
                    embedding_all = torch.cat(
                        [embedding, continuous_transformed], dim=1
                    )
            else:
                embedding_all = continuous_transformed
            return embedding_all.float()

        # Dimension augmentation for known future data
        elif cont is not None and is_multivariate:
            # Continuous process
            continuous = cont[:, :, self.n_temporal_categorical_features :]
            cont_emb_continuous = cont_emb[self.n_temporal_categorical_features]
            cont_bias_continuous = cont_bias[self.n_temporal_categorical_features :]
            continuous_transformed = (
                torch.mul(continuous.unsqueeze(-1), cont_emb_continuous)
                + cont_bias_continuous
            )
            # Categorical process
            categorical = cont[:, :, : self.n_temporal_categorical_features].int()
            embedding_representation = []
            if self.n_temporal_categorical_features > 0:
                for i in range(self.n_temporal_categorical_features):
                    embedding_representation.append(
                        self.embedding_list_future[i](categorical[:, :, i])
                    )
                    embedding = F.gelu(torch.stack(embedding_representation, dim=1))
                    embedding = embedding.permute((0, 2, 1, 3))
                    embedding_all = torch.cat(
                        [embedding, continuous_transformed], dim=2
                    )
            else:
                embedding_all = continuous_transformed
            return embedding_all.float()
        return None

    def forward(
        self,
        target_inp: Tensor,
        stat_exog: Tensor | None = None,
        multi_exog: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Apply tft embedding from the inputs static feature and multivariates features
        format the input


        Args:
            target_inp (Tensor): trend associated to the target
            stat_exog (Tensor | None, optional): static features. Defaults to None.
            multi_exog (Tensor | None, optional): multivariates temporal features in futur data. Defaults to None.

        Returns:
            tuple[Tensor, Tensor, Tensor]: [s_inp, k_inp, target_inp]
            s_inp -> static features embedding
            k_inp -> multivariates temporal features embedding
            target_inp -> target's trend embedding
        """
        # temporal/static categorical/continuous known/observed input
        # tries to get input, if fails returns None

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        stat_exog = stat_exog[:, :] if stat_exog is not None else None

        s_inp = self._apply_embedding(
            cont=stat_exog,
            cont_emb=self.stat_exog_embedding_vectors,
            cont_bias=self.stat_exog_embedding_bias,
            is_stat_exog=True,
        )

        k_inp = self._apply_embedding(
            cont=multi_exog,
            cont_emb=self.multi_exog_embedding_vectors,
            cont_bias=self.multi_exog_embedding_bias,
            is_multivariate=True,
        )

        # Temporal observed targets
        target_inp = torch.matmul(
            target_inp.unsqueeze(3).unsqueeze(4),
            self.tgt_embedding_vectors.unsqueeze(1),
        ).squeeze(3)
        target_inp = target_inp + self.tgt_embedding_bias

        return s_inp, k_inp, target_inp
