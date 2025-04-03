import numpy as np


def create_features(df):
    df['hour_of_day_offer_sin'] = np.sin(2 * np.pi * df['hour_of_day_offer']/24)
    df['hour_of_day_offer_cos'] = np.cos(2 * np.pi * df['hour_of_day_offer']/24)
    df['hour_of_day_offer_select_sin'] = np.sin(2 * np.pi * df['hour_of_day_offer_select']/24)
    df['hour_of_day_offer_select_cos'] = np.cos(2 * np.pi * df['hour_of_day_offer_select']/24)
    df['hour_of_day_auction_sin'] = np.sin(2 * np.pi * df['hour_of_day_auction']/24)
    df['hour_of_day_auction_cos'] = np.cos(2 * np.pi * df['hour_of_day_auction']/24)

    df['model_avg_price'] = df.groupby(['make','generic_model'])['guide_price_auction'].transform('mean')
    df['model_std_price'] = df.groupby(['make','generic_model'])['guide_price_auction'].transform('std')
    df['make_avg_price'] = df.groupby('generic_model')['guide_price_auction'].transform('mean')
    df['make_std_price'] = df.groupby('generic_model')['guide_price_auction'].transform('std')

    df['price_to_make_avg_ratio'] = df['guide_price_auction'] / df['make_avg_price']

    return df
