import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.layers.pooling_wrapper import (
    PoolingWrapper,
)
from models.layers.der import DenseNormalGamma


class MinkLocEvd(torch.nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooling: PoolingWrapper,
        normalize_embeddings: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}  # For pooling statistics
        self.uncertainty_head = DenseNormalGamma(
            n_input=self.pooling.output_dim, embedding_dim=self.pooling.output_dim
        )

    def forward(self, batch):
        x = ME.SparseTensor(batch["features"], coordinates=batch["coords"])
        x = self.backbone(x)

        # Apply pooling to get a global descriptor (embedding)
        # x will be (batch_size, self.pooling.output_dim)
        x = self.pooling(x)
        if hasattr(self.pooling, "stats"):
            self.stats.update(self.pooling.stats)

        assert (
            x.dim() == 2
        ), f"Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions."
        assert x.shape[1] == self.pooling.output_dim, (
            f"Output tensor has: {x.shape[1]} channels. "
            f"Expected: {self.pooling.output_dim}"
        )

        gamma, nu, alpha, beta = self.uncertainty_head(x)
        # Each of these (gamma, nu, alpha, beta) will have shape (batch_size, self.pooling.output_dim)

        # 'global' embedding: this is the mean (gamma) of the evidential distribution.
        if self.normalize_embeddings:
            emb = F.normalize(gamma, dim=1)
        else:
            emb = gamma

        # Return the main embedding ('global') and all NIG parameters.
        # 'global' will be used for computing distances in the triplet loss.
        # gamma, nu, alpha, beta will be used for evidential loss components.
        return {"global": emb, "gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    def print_info(self):
        print("Model class: MinkLocEvd (MinkLoc with Evidential Regression Head)")
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f"Total parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f"Backbone: {type(self.backbone).__name__} #parameters: {n_params}")
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f"Pooling method: {self.pooling.pool_method}   #parameters: {n_params}")
        n_params = sum(
            [param.nelement() for param in self.uncertainty_head.parameters()]
        )
        print(f"Uncertainty Head (DenseNormalGamma) #parameters: {n_params}")
        print("# channels from the backbone: {}".format(self.pooling.in_dim))
        print("# output channels (embedding_dim): {}".format(self.pooling.output_dim))
        print(f"Embedding normalization: {self.normalize_embeddings}")
