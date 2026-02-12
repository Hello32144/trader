os.environ['TF_USE_LEGACY_KERAS'] = '1'
from transformers import pipeline
import tensorflow_decision_forests as tfdf
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import math
import requests
#USE POLYGON.IO
model_name = "yiyanghkust/finbert-tone"
finbert = pipeline("sentiment-analysis", model = model_name)
#headlines = []
#results = finbert(headlines)
#for i, result in enumerate(results):
    #label = result['label']
    #confidence = result['score']
    #sentiment score = 0
    #if label == 'Positive' : sentiment_score = confidence
    #elif label == 'Negative' : sentiment_score = -confidence

    #print(f"Headline: {headlines[i]}"")
    #print(f"Decision {label}, Value {sentiment_score:.4f}\n")
