import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

feature_cols = [
"Volatility_50", 
    "ROC_10", 
    "Volume_ROC", 
    "Return_1d", 
    "CCI_20", 
    "High_Low_Pct", 
    "MOM_5", 
    "Price_to_SMA50", 
    "SMA50_to_SMA200", 
    "SMA20_to_SMA50",
    "SMA10_to_SMA20",
    "Volume_Ratio",
    "DayOfWeek_Sin",
    "ROC_5",
    "MFI_14",
    "Volatility_20",
    "Price_to_SMA200",
    "MACD_SIGNAL"
]
#loads the data from fetcher
print("data is loading")
df = pd.read_csv("training_data_1d.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

#0 means hold because nothing huge happened
df["target_1d"] = 0
df.loc[df['target_return_1d'] > df['ATR_Pct'] * 1.25, "target_1d"] = 1
df.loc[df['target_return_1d'] < -df['ATR_Pct'] * 1.25, "target_1d"] = -1


print(df["target_1d"].value_counts())

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
sample_weights = np.ones(len(y_train))
#weights to tell model that some signals are stornger rather than ablanced
sample_weights[y_train == 1] = 12
sample_weights[y_train == -1] = 12
sample_weights[y_train == 0] = 1

model = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.015,
    min_samples_leaf=25,
    max_leaf_nodes=40,
    l2_regularization=2,
    max_bins=255,
    n_iter_no_change=25,
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
