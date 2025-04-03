import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, random_split

from data.dataset import CarDataset
from evaluation import evaluate_model
from optim.price_optimiser import PriceOptimizer
from train import train_model

df = pd.read_csv('/kaggle/input/processed.csv')
df = df.drop(['pricing_machine_gp_key'], axis=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = CarDataset(df)

total_size = len(full_dataset)
test_size = int(0.2 * total_size)
train_size = total_size - test_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Data Loaders
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model training
model, likelihood1, likelihood2, likelihood3 = train_model(full_dataset, train_loader)

# Model Evaluation
evaluate_model(model, likelihood1, likelihood2, likelihood3, full_dataset, df.iloc[0])

# Price Optimisation
bounds = np.array([[500, 30000]]).reshape(-1, 2)
price_optimiser = PriceOptimizer(model, likelihood1, likelihood2, likelihood3, full_dataset, device, bounds)
X_sample, Y_sample = price_optimiser.optimize(n_iter=50)

best_price, best_prob = price_optimiser.get_best()
print(f"Best price: {best_price}")
print(f"Best offer-select probability: {best_prob}")


