import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

feature_cols = [
    'Stoch_D',
    'MOM_10',
    'SMA10_to_SMA20',
    'MFI_14',
    'Price_to_SMA50',
    'High_Low_Pct',
    'BB_Width',
    'OBV_EMA',
    'Price_to_SMA200',
    'RSI_5',
    'ROC_20',
    'Return_1d',
    'RSI_21'
]
#loads the data from fetcher
print("data is loading")
df = pd.read_csv("training_data_1d.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

#0 means hold because nothing huge happened
df["target_1d"] = 0
df.loc[df['target_return_1d'] > df['ATR_Pct'] * 1.5, "target_1d"] = 1
df.loc[df['target_return_1d'] < -df['ATR_Pct'] * 1.5, "target_1d"] = -1

split_date = df["Datetime"].quantile(0.8)
train_df = df[df["Datetime"] < split_date].copy()
test_df = df[df["Datetime"] >= split_date].copy()

X_train = train_df[feature_cols].values
y_train = train_df["target_1d"].values
X_test = test_df[feature_cols].values
y_test = test_df["target_1d"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sample_weights = compute_sample_weight('balanced', y_train)



print("TRAINING MODEL")
model = HistGradientBoostingClassifier(
    max_iter=3000,
    max_depth=4,
    learning_rate=0.01,
    min_samples_leaf=150,
    l2_regularization = 20,

    max_bins=255,
    n_iter_no_change= 100,
    validation_fraction=0.1,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train, sample_weight=sample_weights)


y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


joblib.dump(model, "model_1d.pkl")
joblib.dump(scaler, "scaler_1d.pkl")
print("SAVED FILES")
