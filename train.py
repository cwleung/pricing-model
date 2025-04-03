import gpytorch
import torch

from model.state_space_svgp import StateSpaceSVGP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(full_dataset, train_loader):
    model = StateSpaceSVGP(
        nn1_dims=full_dataset.X_nn1.shape[1],
        nn2_dims=full_dataset.X_nn2.shape[1],
        nn3_dims=full_dataset.X_nn3.shape[1],
        offer_gp_dims=full_dataset.X_offer.shape[1],
        offer_select_gp_dims=full_dataset.X_offer_select.shape[1],
        auction_gp_dims=full_dataset.X_auction.shape[1]
    )

    model.to(device)
    likelihood1 = gpytorch.likelihoods.BernoulliLikelihood().to(device)
    likelihood2 = gpytorch.likelihoods.BernoulliLikelihood().to(device)
    likelihood3 = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood1.parameters()},
        {'params': likelihood2.parameters()},
        {'params': likelihood3.parameters()}
    ], lr=0.01)

    mll1 = gpytorch.mlls.VariationalELBO(likelihood1, model.gp1, num_data=len(train_loader.dataset))
    mll2 = gpytorch.mlls.VariationalELBO(likelihood2, model.gp2, num_data=len(train_loader.dataset))
    mll3 = gpytorch.mlls.VariationalELBO(likelihood3, model.gp3, num_data=len(train_loader.dataset))

    model.train()
    likelihood1.train()
    likelihood2.train()
    likelihood3.train()

    model.train()
    for epoch in range(100):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            dist1, dist2, dist3 = model(
                batch['X_nn1'],
                batch['X_nn2'],
                batch['X_nn3'],
                batch['X_gp_offer'],
                batch['X_gp_offer_select'],
                batch['X_gp_auction']
            )
            y = batch['y'].squeeze().float()
            mask2 = batch['mask_offer_select'].squeeze()
            mask3 = batch['mask_auction'].squeeze()
            combined_mask = torch.logical_or(mask2.bool(), mask3.bool()).float()

            loss1 = -mll1(dist1, combined_mask)
            loss2 = -mll2(dist2, y) * combined_mask
            loss3 = -mll3(dist3, y) * mask3

            loss = loss1.sum() + loss2.sum() + loss3.sum()

            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    return model, likelihood1, likelihood2, likelihood3