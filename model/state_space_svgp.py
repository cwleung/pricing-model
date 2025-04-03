import torch
import gpytorch
import torch.nn as nn

from nn_mean import NeuralNetworkMean
from svgp import SVGP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StateSpaceSVGP(nn.Module):
    def __init__(self, nn1_dims, nn2_dims, nn3_dims, offer_gp_dims, offer_select_gp_dims, auction_gp_dims):
        super().__init__()
        self.nn_mean1 = NeuralNetworkMean(nn1_dims, 1)
        self.nn_mean2 = NeuralNetworkMean(nn2_dims + 1, 1)
        self.nn_mean3 = NeuralNetworkMean(nn3_dims + 1, 1)

        # layer norm
        self.layer_norm1 = nn.LayerNorm(nn1_dims)
        self.layer_norm2 = nn.LayerNorm(nn2_dims+1)
        self.layer_norm3 = nn.LayerNorm(nn3_dims+1)

        self.gp1 = SVGP(offer_gp_dims, mean_type='linear', kernel_type='rbf')
        self.gp2 = SVGP(offer_select_gp_dims+1, mean_type='linear', kernel_type='matern')
        self.gp3 = SVGP(auction_gp_dims+1, mean_type='linear', kernel_type='matern')

        self.to(device)

    def forward(self,x_nn1,x_nn2,x_nn3,x_offer,x_offer_select,x_auction):
        with gpytorch.settings.cholesky_jitter(1e-4):
            with gpytorch.settings.max_cholesky_size(2000):
                x_nn1_normed = self.layer_norm1(x_nn1)
                nn_mean1 = self.nn_mean1(x_nn1_normed)

                x_nn2_normed = self.layer_norm2(torch.cat([nn_mean1, x_nn2], dim=-1))
                nn_mean2 = self.nn_mean2(x_nn2_normed)

                x_nn3_normed = self.layer_norm3(torch.cat([nn_mean2, x_nn3], dim=-1))
                nn_mean3 = self.nn_mean3(x_nn3_normed)

                output1 = self.gp1(x_offer)
                mean1 = nn_mean1.squeeze(-1)
                covar1 = output1.covariance_matrix + torch.eye(output1.covariance_matrix.size(-1)).to(device) * 1e-9
                dist1 = gpytorch.distributions.MultivariateNormal(mean1, covar1)

                output2 = self.gp2(torch.cat([x_offer_select, nn_mean2],dim=-1))
                mean2 = nn_mean2.squeeze(-1)
                covar2 = output2.covariance_matrix + torch.eye(output2.covariance_matrix.size(-1)).to(device) * 1e-9
                dist2 = gpytorch.distributions.MultivariateNormal(mean2, covar2)

                output3 = self.gp3(torch.cat([x_auction, nn_mean3], dim=-1))
                mean3 = nn_mean3.squeeze(-1)
                covar3 = output3.covariance_matrix + torch.eye(output3.covariance_matrix.size(-1)).to(device) * 1e-9
                dist3 = gpytorch.distributions.MultivariateNormal(mean3, covar3)

                return dist1, dist2, dist3
