import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import create_features
import torch


data = {
    "pricing_machine_gp_key": "7bc989e0-69f6-436e-bc68-f118fdc464ce",
    "guide_price_offer": 500.0,
    "mileage_offer": 50000.0,
    "timestamp_offer": "2024-03-03 13:02:03",
    "guide_price_offer_select": 500.0,
    "mileage_offer_select": 50000.0,
    "timestamp_offer_select": "2024-03-03 13:02:03",
    "guide_price_auction": 500.0,
    "mileage_auction": 50000.0,
    "timestamp_auction": "2024-03-05 19:00:14",
    "gross_sold": 1,
    "make": "Volkswagen",
    "generic_model": "GOLF",
    "fuel": "Petrol",
    "body": "Hatchback",
    "transmission": "Manual",
    "previous_keepers_plt": 3.0,
    "manufacture_year": 2015,
    "offer_accept": True,
    "auction_accept": True,
    "price_diff_auction": 0,
    "year_used": 9.0,
    "mileage_per_year": 5555.55,
    "high_end": 0.0,
    "hour_of_day_offer": 13,
    "day_of_year_offer": 63,
    "hour_of_day_offer_select": 13,
    "day_of_year_offer_select": 63,
    "hour_of_day_auction": 19,
    "day_of_year_auction": 65,
    "time_diff_offer_select_minutes_log": 0.0,
    "time_diff_auction_minutes_log": 8.083076519401347,
    "hour_of_day_offer_sin": -0.2588190451025208,
    "hour_of_day_offer_cos": -0.9659258262890683,
    "hour_of_day_offer_select_sin": -0.2588190451025208,
    "hour_of_day_offer_select_cos": -0.9659258262890683,
    "hour_of_day_auction_sin": -0.9659258262890684,
    "hour_of_day_auction_cos": 0.2588190451025203,
    "model_avg_price": 11285.716555183946,
    "model_std_price": 6849.891796165352,
    "make_avg_price": 11373.690124529394,
    "make_std_price": 7820.051138066778,
    "price_to_make_avg_ratio": 0.938833385039301
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, likelihood1, likelihood2, likelihood3, full_dataset, data):
    valid_data = []
    price_range = np.arange(500, 30001, 500)
    for i in price_range:
        data['guide_price_offer'] = i
        data['guide_price_offer_select'] = i
        data['guide_price_auction'] = i
        valid_data.append(data.copy())

    valid_data = pd.DataFrame(valid_data)
    X_valid = create_features(valid_data)
    valid_dataset = full_dataset.transform(X_valid)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    pred_mean = []
    pred_std = []
    for idx, bt in enumerate(valid_data_loader):
        model.eval()
        likelihood1.eval()
        likelihood2.eval()
        likelihood3.eval()
        bt = {key: value.to(device) for key, value in bt.items()}

        dist1, dist2, dist3 = model(
            bt['X_nn1'],
            bt['X_nn2'],
            bt['X_nn3'],
            bt['X_gp_offer'],
            bt['X_gp_offer_select'],
            bt['X_gp_auction']
        )

        pred1 = likelihood1(dist1)
        pred2 = likelihood2(dist2)
        pred3 = likelihood3(dist3)
        pred_mean.append((pred1.mean.item(), pred2.mean.item(), pred3.mean.item()))
        pred_std.append((np.sqrt(pred1.variance.item()), np.sqrt(pred2.variance.item()), np.sqrt(pred3.variance.item())))

    preds_array = np.array(pred_mean)
    column_sums = preds_array.sum(axis=0)
    normalized_preds = pred_mean / column_sums

    plt.figure(figsize=(10, 6))
    plt.plot(price_range, np.array(normalized_preds)[:, 1], label='Offer Select', linewidth=2)
    plt.plot(price_range, np.array(normalized_preds)[:, 2], label='Auction', linewidth=2)
    plt.fill_between(price_range, np.array(normalized_preds)[:, 1] - np.array(pred_std)[:, 1], np.array(normalized_preds)[:, 1] + np.array(pred_std)[:, 1], alpha=0.2)
    plt.fill_between(price_range, np.array(normalized_preds)[:, 2] - np.array(pred_std)[:, 2], np.array(normalized_preds)[:, 2] + np.array(pred_std)[:, 2], alpha=0.2)
    plt.xlabel('Price Range')
    plt.ylabel('Prediction')
    plt.title('Predictions Across Different Price Ranges')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()