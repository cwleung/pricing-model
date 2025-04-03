import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

categorical_columns = ['make','generic_model', 'fuel', 'body', 'transmission']
numerical_columns = ['previous_keepers_plt', 'year_used', 'mileage_per_year', 'high_end', 'model_avg_price', 'model_std_price', 'make_avg_price', 'make_std_price']

def handle_imbalance(X, y):
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = ImbPipeline(steps=steps)
    return over.fit_resample(X, y)

def create_preprocessing_pipeline(df):

    numerical_columns = [col for col in df.columns if df[col].dtype == 'float64' or df[col].dtype == 'int64']
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    numeric_transformer = Pipeline(steps=[
        # ('yeo-johnson', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder()),
        ('svd', TruncatedSVD(n_components=50))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
    )

    return preprocessor

def process_data(df):
    X = df.drop(['gross_sold'], axis=1)
    y = df['gross_sold']

    preprocessor = create_preprocessing_pipeline()
    X_processed = preprocessor.fit_transform(X, y)
    return X_processed, y, preprocessor



# NN 1 Mean function
nn1_columns = numerical_columns + categorical_columns + ['guide_price_offer', 'mileage_offer', 'hour_of_day_offer_sin', 'hour_of_day_offer_cos']
# NN 2 Mean function
nn2_columns = numerical_columns + categorical_columns + ['guide_price_offer_select', 'hour_of_day_offer_select_sin', 'hour_of_day_offer_select_cos','time_diff_offer_select_minutes_log']
# NN 3 Mean function
nn3_columns = numerical_columns + categorical_columns + ['guide_price_auction', 'time_diff_auction_minutes_log', 'hour_of_day_auction_sin', 'hour_of_day_auction_cos']
# layer 1 GP~(NN_1(x), Kernel_1)
offer_gp_columns = numerical_columns + ['guide_price_offer', 'mileage_offer', 'hour_of_day_offer']
# Layer 2 GP~(NN_2(x), Kernel_2)
offer_select_gp_columns = numerical_columns + ['guide_price_offer_select', 'mileage_offer_select', 'hour_of_day_offer_select','time_diff_offer_select_minutes_log']
# Layer 3 GP~(NN_3(x), Kernel_3)
auction_gp_columns = numerical_columns + ['guide_price_auction', 'mileage_auction', 'hour_of_day_auction','time_diff_auction_minutes_log']

class CarDataset(Dataset):
    def __init__(self, df, fit=True):
        if fit:
            self.preprocessor_nn1 = create_preprocessing_pipeline(df[nn1_columns])
            self.preprocessor_nn2 = create_preprocessing_pipeline(df[nn2_columns])
            self.preprocessor_nn3 = create_preprocessing_pipeline(df[nn3_columns])
            self.preprocessor_offer = create_preprocessing_pipeline(df[offer_gp_columns])
            self.preprocessor_offer_select = create_preprocessing_pipeline(df[offer_select_gp_columns])
            self.preprocessor_auction = create_preprocessing_pipeline(df[auction_gp_columns])

            self.X_nn1 = self.preprocessor_nn1.fit_transform(df[nn1_columns])
            self.X_nn2 = self.preprocessor_nn2.fit_transform(df[nn2_columns])
            self.X_nn3 = self.preprocessor_nn3.fit_transform(df[nn3_columns])

            self.X_offer = self.preprocessor_offer.fit_transform(df[offer_gp_columns])
            self.X_offer_select = self.preprocessor_offer_select.fit_transform(df[offer_select_gp_columns])
            self.X_auction = self.preprocessor_auction.fit_transform(df[auction_gp_columns])

        self.y = df['gross_sold'].values
        self.mask_offer_select = df['offer_accept'].values
        self.mask_auction = df['auction_accept'].values

    def transform(self, df):
        X_nn1 = self.preprocessor_nn1.transform(df[nn1_columns])
        X_nn2 = self.preprocessor_nn2.transform(df[nn2_columns])
        X_nn3 = self.preprocessor_nn3.transform(df[nn3_columns])
        X_offer = self.preprocessor_offer.transform(df[offer_gp_columns])
        X_offer_select = self.preprocessor_offer_select.transform(df[offer_select_gp_columns])
        X_auction = self.preprocessor_auction.transform(df[auction_gp_columns])

        transformed_dataset = CarDataset(df, fit=False)
        transformed_dataset.X_nn1 = X_nn1
        transformed_dataset.X_nn2 = X_nn2
        transformed_dataset.X_nn3 = X_nn3
        transformed_dataset.X_offer = X_offer
        transformed_dataset.X_offer_select = X_offer_select
        transformed_dataset.X_auction = X_auction

        return transformed_dataset

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = {
            'X_nn1': torch.FloatTensor(self.X_nn1[idx]),
            'X_nn2': torch.FloatTensor(self.X_nn2[idx]),
            'X_nn3': torch.FloatTensor(self.X_nn3[idx]),
            'X_gp_offer': torch.FloatTensor(self.X_offer[idx]),
            'X_gp_offer_select': torch.FloatTensor(self.X_offer_select[idx]),
            'X_gp_auction': torch.FloatTensor(self.X_auction[idx]),
            'mask_offer_select': torch.FloatTensor([self.mask_offer_select[idx]]),
            'mask_auction': torch.FloatTensor([self.mask_auction[idx]]),
            'y': torch.FloatTensor([self.y[idx]])
        }
        return data
