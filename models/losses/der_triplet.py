import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


def evidential_nll_loss(gamma, nu, alpha, beta, y_true):
    """
    Computes the negative log-likelihood loss for the evidential regression model.
    Args:
        gamma (torch.Tensor): The mean prediction from the model.
        nu (torch.Tensor): The precision parameter for the normal distribution.
        alpha (torch.Tensor): The shape parameter for the gamma distribution.
        beta (torch.Tensor): The scale parameter for the gamma distribution.
        y_true (torch.Tensor): The true target values.
    Returns:
        torch.Tensor: The computed negative log-likelihood loss.
    """
    error_sq = (y_true - gamma) ** 2

    # For numerical stability, ensure nu and beta are strictly positive
    nu = nu.clamp(min=1e-6)
    alpha = alpha.clamp(min=1.0 + 1e-6)  # Ensure alpha > 1
    beta = beta.clamp(min=1e-6)

    loss_nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(beta)
        + (alpha + 0.5) * torch.log(beta + 0.5 * nu * error_sq)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return loss_nll


def evidential_regularization_loss(gamma, nu, alpha, y_true):
    """
    Computes the regularization loss for the evidential regression model.
    Args:
        gamma (torch.Tensor): The mean prediction from the model.
        nu (torch.Tensor): The precision parameter for the normal distribution.
        alpha (torch.Tensor): The shape parameter for the gamma distribution.
        y_true (torch.Tensor): The true target values.
    Returns:
        torch.Tensor: The computed regularization loss.
    """
    # L_R = |y_i - gamma_i| * (2*nu_i + alpha_i)
    absolute_error = torch.abs(y_true - gamma)

    # Ensure nu and alpha are strictly positive for phi calculation
    nu = nu.clamp(min=1e-6)
    alpha = alpha.clamp(min=1.0 + 1e-6)  # Ensure alpha > 1

    total_evidence_term = 2 * nu + alpha
    loss_reg = absolute_error * total_evidence_term
    return loss_reg


class DerTripletMarginLoss(BaseMetricLossFunction):
    """
    Implements the standard Triplet Margin Loss integrating Deep Evidential Regression loss function

    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        der_lambda=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.der_lambda = der_lambda
    
    def forward(
        self,
        embeddings,
        labels,
        indices_tuple,
        gamma,
        nu,
        alpha,
        beta,
        ref_emb=None,
        ref_labels=None,
    ):
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)

        # Chiamiamo il nostro metodo compute_loss con tutti i parametri necessari
        loss_dict = self.compute_loss(
            embeddings,
            labels,
            indices_tuple,
            gamma,
            nu,
            alpha,
            beta,
        )
        
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        
        return loss_dict


    def compute_loss(
        self,
        embeddings,
        labels,
        indices_tuple,
        gamma,
        nu,
        alpha,
        beta,
    ):

        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, t_per_anchor=self.triplets_per_anchor
        )

        anchor_idx, positive_idx, negative_idx = indices_tuple

        if len(anchor_idx) == 0:
            return self.zero_losses()

        # Extract NIG parameters for anchor, positive, and negative samples
        gamma_a, nu_a, alpha_a, beta_a = (
            gamma[anchor_idx],
            nu[anchor_idx],
            alpha[anchor_idx],
            beta[anchor_idx],
        )
        gamma_p, nu_p, alpha_p, beta_p = (
            gamma[positive_idx],
            nu[positive_idx],
            alpha[positive_idx],
            beta[positive_idx],
        )
        gamma_n, nu_n, alpha_n, beta_n = (
            gamma[negative_idx],
            nu[negative_idx],
            alpha[negative_idx],
            beta[negative_idx],
        )

        # NLL(Anchor, Positive) e NLL(Anchor, Negative) are the negative log-likelihood losses
        # for the Anchor with respect to the Positive and Negative samples, respectively.

        # NLL(Anchor, Positive): how much the Anchor parameters "explain" the embedding of the Positive.
        # NLL(Anchor, Negative): how much the Anchor parameters "explain" the embedding of the Negative.

        # y_true for the NLL is the embedding (gamma) of the other element in the pair.
        nll_ap = evidential_nll_loss(gamma_a, nu_a, alpha_a, beta_a, gamma_p)
        nll_an = evidential_nll_loss(gamma_a, nu_a, alpha_a, beta_a, gamma_n)

        # As nll_ap and nll_an are now tensors of shape (num_triplets, embedding_dim),
        # we need to take the mean over the embedding_dim to have a scalar cost per triplet.
        nll_ap_per_triplet = torch.mean(nll_ap, dim=1)
        nll_an_per_triplet = torch.mean(nll_an, dim=1)

        # The component of the triplet loss is: loss_nll_tr = [NLL_AP - NLL_AN + margin]
        # we want to minimize the NLL for the Anchor-Positive pair and maximize the NLL for the Anchor-Negative pair.
        loss_nll_tr = nll_ap_per_triplet - nll_an_per_triplet + self.margin

        # L_R^TR = [phi_A * d_ap - phi_A * d_an + m]_+
        # L_R^TR = [phi_P * d_ap - phi_N * d_an + m]_+

        # Compute the evidence Phi for anchor, positive, and negative.
        # The evidences are (2*nu + alpha). Since nu and alpha are of shape (num_triplets, embedding_dim),
        # we take the mean to have a scalar evidence for each triplet.
        phi_anchor = (2 * nu_a + alpha_a).mean(dim=1)
        phi_positive = (2 * nu_p + alpha_p).mean(dim=1)
        phi_negative = (2 * nu_n + alpha_n).mean(dim=1)

        # Compute the euclidean distances between the embeddings (gamma).
        # embeddings here is the global parameter returned by the model, which is gamma (normalized or not).
        # We use it because it is the one that will be compared in standard PR metrics.
        mat = self.distance(embeddings)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]

        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        # Regularization term based on the evidence of the Anchor
        loss_reg_a = (phi_anchor * ap_dists) - (phi_anchor * an_dists) + self.margin

        # Regularization term based on the evidence of Positive and Negative
        loss_reg_pn = (
            (phi_positive * ap_dists) - (phi_negative * an_dists) + self.margin
        )

        # Sum the two regularization components (L_R^TR = L_R,A^TR + L_R,P,N^TR)
        loss_reg_tr = loss_reg_a + loss_reg_pn

        # ------ Total Loss DER-TR ------
        # L_DER^TR = phi_ML^TR + lambda * phi_R^TR (dagli appunti)

        total_violation = loss_nll_tr + self.der_lambda * loss_reg_tr

        if self.smooth_loss:
            loss = torch.nn.functional.softplus(total_violation)
        else:
            loss = torch.nn.functional.relu(total_violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()
