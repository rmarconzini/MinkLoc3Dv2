import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNormalGamma(nn.Module):
    def __init__(self, n_input: int, embedding_dim: int):
        """
        Layer to predict the four parameters of the Normal-Inverse Gamma (NIG) distribution.

        Args:
            n_input (int): Dimension of the input feature vector.
            embedding_dim (int): The dimensionality of the embedding for which NIG parameters are predicted.
                                 This will also be the dimensionality of gamma, nu, alpha, beta.
        """
        super(DenseNormalGamma, self).__init__()
        self.n_in = n_input
        self.embedding_dim = embedding_dim
        # The output layer will produce 4 * embedding_dim values,
        # which will then be split into 4 tensors of shape (batch_size, embedding_dim)
        self.linear = nn.Linear(self.n_in, 4 * self.embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor from the backbone/pooling, shape (batch_size, n_input).

        Returns:
            tuple: (gamma, nu, alpha, beta), each a torch.Tensor of shape (batch_size, embedding_dim).
        """

        # Ensure input is 2D (batch_size, features)
        assert (
            x.dim() == 2
        ), f"Expected 2-dimensional input (batch_size, features), got {x.dim()} dimensions."

        # Pass through the linear layer
        # Output shape: (batch_size, 4 * embedding_dim)
        x_out = self.linear(x)

        # Split the output into the four NIG parameters
        # Each parameter will have shape (batch_size, embedding_dim)
        gamma, log_nu, log_alpha, log_beta = torch.split(
            x_out, self.embedding_dim, dim=1
        )

        # Apply activations to ensure correct ranges for nu, alpha, beta
        # Adding a small epsilon for numerical stability
        nu = F.softplus(log_nu) + 1e-6  # nu > 0
        alpha = (
            F.softplus(log_alpha) + 1.0
        )  # alpha > 1 (important for E[sigma^2] = beta / (alpha - 1))
        beta = F.softplus(log_beta) + 1e-6  # beta > 0

        return gamma, nu, alpha, beta
