

import os
from transformers import pipeline
import datetime
import numpy as np
import pandas as pd
from massive import RESTClient
from finta import TA
import time




API_KEY = "im not giving u it"
client = RESTClient(API_KEY, pagination=True)


finbert_base = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0, batch_size=64, truncation=True)                
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", device=0, batch_size=64, truncation=True)
models = [finbert_base, finbert_tone]


def get_news(stock, client):
    news_list = []


    for n in client.list_ticker_news(
        ticker=stock,
        published_utc_gte="2024-02-13",
        published_utc_lte="2026-02-13",
        order="asc",
        limit=1000
    ):
        published_time = pd.to_datetime(n.published_utc, utc=True)
        news_list.append({
            "Headline": n.title,
            "Stock": stock,
            "Datetime": published_time
        })


   
    if not news_list: return pd.DataFrame()
   
    df = pd.DataFrame(news_list)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.set_index("Datetime")


    df.index = df.index.floor('1d')
    return df


def get_candles(stock, client):
    stock_data = []


    for a in client.list_aggs(
        stock,
        1,
        "day", # Fetches Daily Data
        from_="2024-02-13",
        to="2026-02-13",
        adjusted=True,
        sort="asc",
        limit=50000
    ):
        stock_data.append({
            "Datetime": pd.to_datetime(a.timestamp, unit="ms", utc=True),
            "Stock": stock,
            "open": a.open,
            "close": a.close,
            "low": a.low,
            "high": a.high,
            "volume": a.volume
        })
   
    df = pd.DataFrame(stock_data)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.set_index("Datetime")
    return df


def get_indicators(df):
    df["RSI_5"] = TA.RSI(df, period=5)
    df["RSI_14"] = TA.RSI(df, period=14)
    df["RSI_21"] = TA.RSI(df, period=21)
   
    df["ROC_5"] = TA.ROC(df, period=5)
    df["ROC_10"] = TA.ROC(df, period=10)
    df["ROC_20"] = TA.ROC(df, period=20)
   
    df["MOM_5"] = TA.MOM(df, period=5)
    df["MOM_10"] = TA.MOM(df, period=10)
   
    df["Williams_R"] = TA.WILLIAMS(df, period=14)
   
    # FIX: STOCH returns a Series, not a DataFrame
    df["Stoch_K"] = TA.STOCH(df, period=14)
    # FIX: STOCHD is a separate function
    df["Stoch_D"] = TA.STOCHD(df, period=14)
   
    df["CCI_20"] = TA.CCI(df, period=20)
    df["MFI_14"] = TA.MFI(df, period=14)
   
    df["ATR_14"] = TA.ATR(df, period=14)
    df["ATR_Pct"] = df["ATR_14"] / df["close"]
    df["High_Low_Pct"] = (df["high"] - df["low"]) / df["close"]
   
    df["Volatility_20"] = df["close"].pct_change().rolling(20).std()
    df["Volatility_50"] = df["close"].pct_change().rolling(50).std()
   
    bb = TA.BBANDS(df, period=20)
    df["BB_Position"] = (df["close"] - bb["BB_LOWER"]) / (bb["BB_UPPER"] - bb["BB_LOWER"])
    df["BB_Width"] = (bb["BB_UPPER"] - bb["BB_LOWER"]) / df["close"]
   
    dc_upper = df["high"].rolling(20).max()
    dc_lower = df["low"].rolling(20).min()
    df["DC_Position"] = (df["close"] - dc_lower) / (dc_upper - dc_lower)
   
    df["Volume_SMA_20"] = df["volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["volume"] / df["Volume_SMA_20"]
    df["Volume_ROC"] = df["volume"].pct_change(5)
   
    df["OBV"] = TA.OBV(df)
    df["OBV_EMA"] = df["OBV"].ewm(span=20).mean()
   
    macd = TA.MACD(df)
    df["MACD"] = macd["MACD"]                
    df["MACD_SIGNAL"] = macd["SIGNAL"]                
   
    df["MACD_HIST"] = macd["MACD"]
   
    df["ADX_14"] = TA.ADX(df, period=14)
   
    sma20 = TA.SMA(df, period=20)
    sma50 = TA.SMA(df, period=50)
    sma200 = TA.SMA(df, period=200)
   
    df["Price_to_SMA20"] = df["close"] / sma20
    df["Price_to_SMA50"] = df["close"] / sma50
    df["Price_to_SMA200"] = df["close"] / sma200
   
    sma10 = TA.SMA(df, period=10)
    df["SMA10_to_SMA20"] = sma10 / sma20
    df["SMA20_to_SMA50"] = sma20 / sma50
    df["SMA50_to_SMA200"] = sma50 / sma200
   
    # RENAMED: 'h' to 'd' for daily data
    df["Return_1d"] = df["close"].pct_change(1)
    df["Return_5d"] = df["close"].pct_change(5) # 1 week
    df["Return_20d"] = df["close"].pct_change(20) # 1 month
   
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
   
    return df


def normalize_results(label, score):
    label = label.lower()
    if label == "positive": return score
    elif label == "negative": return -score
    else: return 0


def ensemble_nlp(headlines):
    scores = []
    for model in models:
        results = model(headlines)
        score = [normalize_results(result["label"], result["score"]) for result in results]
        scores.append(score)
    return np.mean(np.array(scores), axis=0)


def get_target_1d(df):
    df = df.sort_values(by=['Stock', 'Datetime'])


    df['target_return_1d'] = df.groupby('Stock')['close'].pct_change(-1)


    df['target_1d'] = (df['target_return_1d'] > 0.007).astype(int)
    df['target_1d'] = (df['target_return_1d'] < 0.007).astype(int)
    return df


# STOCKS
stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
    "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "JPM", "BAC", "WFC", "V", "MA",
    "UNH", "JNJ", "PFE", "WMT", "HD", "PG", "KO", "DIS", "XOM", "CVX",
    "BA", "CAT", "GE", "NKE", "SBUX", "MCD", "T", "VZ", "COST", "TMUS"
]




feature_cols = [
    'RSI_5', 'RSI_14', 'RSI_21', 'Stoch_K', 'Stoch_D', 'Williams_R',
    'CCI_20', 'MFI_14', 'ROC_5', 'ROC_10', 'ROC_20', 'MOM_5', 'MOM_10',
    'ATR_Pct', 'BB_Width', 'BB_Position', 'High_Low_Pct',
    'Volatility_20', 'Volatility_50', 'Volume_Ratio', 'Volume_ROC',
    'OBV_EMA', 'ADX_14', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
    'DC_Position', 'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_SMA200',
    'SMA10_to_SMA20', 'SMA20_to_SMA50', 'SMA50_to_SMA200',
    'Return_1d', 'Return_5d', 'Return_20d',
    'DayOfWeek_Sin', 'DayOfWeek_Cos'
]


all_data = []


for stock in stocks:
    print(f"Calculating {stock}")
    news_df = get_news(stock, client)
    price_df = get_candles(stock, client)
   
    if price_df.empty: continue


    price_df = get_indicators(price_df)
   
    price_df_lagged = price_df[feature_cols].shift(1)
    price_df_lagged["Stock"] = price_df["Stock"]
    price_df_lagged["close"] = price_df["close"]
    price_df_lagged = price_df_lagged.iloc[1:]
   
    if not news_df.empty:
        headlines = news_df["Headline"].tolist()
        scores = ensemble_nlp(headlines)
        news_df["Scores"] = scores
        news_df = news_df.groupby([news_df.index, 'Stock']).agg({'Scores': 'mean'}).reset_index(level=1)
       
        master_df = pd.merge(
            price_df_lagged.reset_index(),
            news_df.reset_index(),
            on=['Datetime', 'Stock'],
            how='left'
        )
        master_df = master_df.set_index('Datetime')
        master_df["Scores"] = master_df["Scores"].fillna(0)
    master_df = get_target_1d(master_df)
    master_df = master_df.dropna()
    all_data.append(master_df)


final_df = pd.concat(all_data, ignore_index=False)
final_df = final_df.reset_index()

final_df.to_csv('training_data_1d.csv', index=False)


