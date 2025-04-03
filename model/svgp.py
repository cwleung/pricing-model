import torch
import gpytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, num_dims, mean_type='constant', kernel_type='rbf'):
        inducing_points = torch.randn(100, num_dims).to(device)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(100)
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        if mean_type == 'constant':
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.LinearMean(num_dims)

        if kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=num_dims,
            )
            self.hourly = gpytorch.kernels.PeriodicKernel()
            kernel = base_kernel + self.hourly
        else:
            kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=num_dims,
            )

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel,
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
