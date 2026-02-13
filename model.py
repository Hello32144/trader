import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

#import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import math
import json
from massive import RESTClient
import requests
import time
from massive.rest.models import (
    TickerNews,
)
headlines = []

client = RESTClient("DiB3myZK1i2Zqw55XlvF6bng2AGedIQ8", pagination = False)
news_list = []

stocks = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
try:
    for stock in stocks:
        news = client.list_ticker_news(stock, limit = 10)
        for item in news:
            headlines.append(item.title)
        time.sleep(15)
except Exception as e:
    print(e)
    time.sleep(60)



model_name = "ProsusAI/finbert"
finbert = pipeline("sentiment-analysis", model = model_name)

results = finbert(headlines)
print(results)
for i, result in enumerate(results):
    label = result['label']
    confidence = result['score']
    sentiment_score = 0
    if label == 'positive' : sentiment_score = confidence
    elif label == 'negative' : sentiment_score = -confidence

    print(f"Headline: {headlines[i]}")
    print(f"Decision {label}, Value {sentiment_score:.4f}\n")
