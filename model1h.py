import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight



feature_cols = [
    'RSI_5', 'RSI_14', 'RSI_21',
    'Stoch_K', 'Stoch_D',
    'Williams_R',
    'CCI_20',
    'MFI_14',
    'ROC_5', 'ROC_10', 'ROC_20',
    'MOM_5', 'MOM_10',
    'ATR_Pct',
    'BB_Width',
    'BB_Position',
    'High_Low_Pct',
    'Volatility_20', 'Volatility_50',
    'Volume_Ratio',
    'Volume_ROC',
    'OBV_EMA',
    'ADX_14',
    'MACD', 'MACD_SIGNAL', 'MACD_HIST',
    'DC_Position',
    'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_SMA200',
    'SMA10_to_SMA20', 'SMA20_to_SMA50', 'SMA50_to_SMA200',
    'Return_1h', 'Return_2h', 'Return_4h',
    'Return_8h', 'Return_12h', 'Return_24h',
    'DayOfWeek_Sin', 'DayOfWeek_Cos',
    'Scores',
]

#loads the data from fetcher
print("data is loading")
df = pd.read_csv("training_data.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

#0 means hold because nothing huge happened 0.001 is the return value that we track
df["target_1h"] = 0
df.loc[df["target_return_1h"] > df['ATR_Pct'] * 1.5, "target_1h"] = 1
df.loc[df["target_return_1h"] < -df['ATR_Pct'] * 1.5, "target_1h"] = -1

split_date = df["Datetime"].quantile(0.8)
train_df = df[df["Datetime"] < split_date].copy()
test_df = df[df["Datetime"] >= split_date].copy()

X_train = train_df[feature_cols].values
y_train = train_df["target_1h"].values
X_test = test_df[feature_cols].values
y_test = test_df["target_1h"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sample_weights = compute_sample_weight('balanced', y_train)



print("TRAINING MODEL")
model = HistGradientBoostingClassifier(
    max_iter=5000,
    max_depth=15,
    learning_rate=0.003,
    min_samples_leaf=20,
    l2_regularization = 0.5,
    validation_fraction=0.1,
    max_bins=255,
    n_iter_no_change= 100,
    random_state=42,
    verbose= 1
)
model.fit(X_train, y_train, sample_weight=sample_weights)


y_pred = model.predict(X_test)
print("report")
print(classification_report(y_test, y_pred))


joblib.dump(model, "model_1h.pkl")
joblib.dump(scaler, "scaler_1h.pkl")
print("SAVED FILES")
