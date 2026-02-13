import os
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import datetime
import numpy as np
import pandas as pd
import math
import json
from massive import RESTClient
import requests
import time
from massive.rest.models import (
    TickerNews,
)
import pandas_ta as ta

import yfinance as yf
price_cache = {}
def get_price_data(stock):
    if stock not in price_cache:
        df = yf.download(stock, interval="1h", period="5d")
        df = df.reset_index()
        price_cache[stock] = df
    return price_cache[stock]

def get_price_date(stock, time):
    df = get_price_data(stock)
    old = df[df["Datetime"] <= time]
    if len(old) == 0:
        return None
    return old.iloc[-1]


def get_price_change(stock, time):
    now = get_price_date(stock, time)
    if now is None:
        return None
    next = time+ datetime.timedelta(hours=1)
    df = get_price_data(stock)
    future = df[df["Datetime"] >= next]

    if len(future) == 0:
        return None
    future = future.iloc[0]
    return (future["Close"]-now["Close"])/now["Close"]


headlines = []

client = RESTClient("DiB3myZK1i2Zqw55XlvF6bng2AGedIQ8", pagination = False)
news_list = []


finbert_base = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

finbert_tone = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone"
)



models = [finbert_base, finbert_tone]

data = []
stocks = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "NFLX"]


try:
    for stock in stocks:
        news = client.list_ticker_news(stock, limit = 100)
        for item in news:
            news_list.append({
                "headline":item.title,
                "date":pd.to_datetime(item.published_utc),
                "ticker":stock})
        time.sleep(15)
except Exception as e:
    print(e)
    time.sleep(60)
    

def normalize(label, score):
    labels = label.lower()

    if label  == 'positive' : return score
    elif label == 'negative' : return -score
    else: return 0
def combine_results(headlines):
    scores=[]
    for model in models:
        results = model(headlines)[0]
        score = normalize(results['label'], results['score'])
        scores.append(score)
    return np.mean(scores)

rows = []

for item in news_list:
    score = combine_results(item["headline"])
    rows.append({
        "headline": item["headline"],
        "date": item["date"],
        "ticker": item["ticker"],
        "score": score
    })

final = []
for item in rows:
    ts = pd.to_datetime(item["date"])
    current_price = get_price_date(item["ticker"], ts)
    future_price = get_price_change(item["ticker"], ts)
    if current_price is None or future_price is None:
        continue

    final.append({
        "headline": item["headline"],
        "date": item["date"],
        "ticker": item["ticker"],
        "score":item['score'],
        "current" : current_price["Close"],
        "1h_return" : future_price
    })
df = pd.DataFrame(final)
df.to_csv("data.csv", index = False)
print("saved")