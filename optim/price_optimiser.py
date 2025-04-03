from scipy.optimize import minimize

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils import create_features


class PriceOptimizer:
    def __init__(self, model, likelihood1, likelihood2, likelihood3, full_dataset, device, bounds, data=None):
        self.model = model
        self.likelihood1 = likelihood1
        self.likelihood2 = likelihood2
        self.likelihood3 = likelihood3
        self.full_dataset = full_dataset
        self.device = device
        self.bounds = np.array(bounds)
        self.X_sample = None
        self.Y_sample = None
        self.X_history = []
        self.Y_history = []

        self.data = data

    def objective_function(self, price, is_auction):
        data_copy = self.data.copy()
        data_copy['guide_price_offer'] = price
        data_copy['guide_price_offer_select'] = price
        data_copy['guide_price_auction'] = price

        df = pd.DataFrame([data_copy])
        X = create_features(df)
        dataset = self.full_dataset.transform(X)
        dataset.mask_offer = np.ones(1, dtype=np.float32)
        dataset.mask_offer_select = np.ones(1, dtype=np.float32)
        dataset.mask_auction = np.ones(1, dtype=np.float32)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for bt in dataloader:
            self.model.eval()
            self.likelihood1.eval()
            self.likelihood2.eval()
            self.likelihood3.eval()
            bt = {key: value.to(self.device) for key, value in bt.items()}

            dist1, dist2, dist3 = self.model(
                bt['X_nn1'], bt['X_nn2'], bt['X_nn3'],
                bt['X_gp_offer'], bt['X_gp_offer_select'], bt['X_gp_auction']
            )

            pred1 = self.likelihood1(dist1)
            pred2 = self.likelihood2(dist2)
            pred3 = self.likelihood3(dist3)

            preds = np.array([pred1.mean.item(), pred2.mean.item(), pred3.mean.item()])
            normalized_preds = preds / preds.sum()

            return normalized_preds[2] if is_auction else normalized_preds[1]

    def optimize(self, n_iter, initial_points=5, is_auction=False):
        dim = len(self.bounds)
        self.X_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(initial_points, dim))
        self.Y_sample = np.array([self.objective_function(x, is_auction=is_auction) for x in self.X_sample])

        self.X_history = [self.X_sample.copy()]
        self.Y_history = [self.Y_sample.copy()]

        for i in range(n_iter):
            result = minimize(
                lambda X: -self.objective_function(X[0], is_auction),
                x0=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=dim),
                bounds=self.bounds,
                method='L-BFGS-B'
            )

            X_next = np.array([result.x])
            Y_next = np.array([self.objective_function(result.x[0], is_auction)])

            self.X_sample = np.vstack((self.X_sample, X_next))
            self.Y_sample = np.append(self.Y_sample, Y_next)

            print(f"Iteration {i + 1}: Price={X_next[0][0]:.2f}, {'Auction' if is_auction else 'Offer-Select'} Prob={Y_next[0]:.3f}")

        return self.X_sample, self.Y_sample

    def get_best(self):
        best_idx = np.argmax(self.Y_sample)
        return self.X_sample[best_idx], self.Y_sample[best_idx]
