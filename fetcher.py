(Get-Command python3.13).Path
price_cache = {}

def get_price_data(stock):
    if stock not in price_cache:
        df = yf.download(stock, interval="1h", period="1mo")
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

        # Technical Indicators
        df["SMA_10"] = TA.SMA(df, 10)
        df["SMA_20"] = TA.SMA(df, 20)
        df["SMA_50"] = TA.SMA(df, 50) 
        df["EMA_12"] = TA.EMA(df, 12)
        df["EMA_26"] = TA.EMA(df, 26)
        df["RSI_14"] = TA.RSI(df, 14)
        
        stoch = TA.STOCH(df)
        if isinstance(stoch, pd.DataFrame):
            if isinstance(stoch.columns, pd.MultiIndex):
                stoch.columns = stoch.columns.get_level_values(0)                
            df["Stoch_K"] = stoch.iloc[:, 0]
            df["Stoch_D"] = stoch.iloc[:, 1]
        else:
            df["Stoch_K"] = stoch
            df["Stoch_D"] = stoch

        df["Williams_R"] = TA.WILLIAMS(df)
        df["ROC"] = TA.ROC(df)
        df["ATR_14"] = TA.ATR(df, 14)

        bb = TA.BBANDS(df)
        df["BB_Upper"] = bb["BB_UPPER"]
        df["BB_Lower"] = bb["BB_LOWER"]
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["Close"]

        df["OBV"] = TA.OBV(df)
        df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()
        df["MFI_14"] = TA.MFI(df)
        
        adx = TA.ADX(df)                
        if isinstance(adx, pd.DataFrame):
            if isinstance(adx.columns, pd.MultiIndex):
                adx.columns = adx.columns.get_level_values(0)
            df["ADX_14"] = adx.iloc[:, 0]
        else:
            df["ADX_14"] = adx
            
        df["CCI_20"] = TA.CCI(df, 20)

        price_cache[stock] = df

    return price_cache[stock]
#outputs a df for all stocks

#grabs data before the set time
def get_price_date(stock, query_time):
    df = get_price_data(stock)
    if df is None: return None
    old = df[df["Datetime"] <= query_time]
    if len(old) == 0: return None
    return old.iloc[-1]
#gets the return for each hour
def get_price_change(stock, query_time):
    now = get_price_date(stock, query_time)
    if now is None: return None
    
    next_time = query_time + datetime.timedelta(hours=1)
    df = get_price_data(stock)
    future = df[df["Datetime"] >= next_time]

    if len(future) == 0: return None
    future = future.iloc[0]
    return (future["Close"] - now["Close"]) / now["Close"]

client = RESTClient("DiB3myZK1i2Zqw55XlvF6bng2AGedIQ8", pagination=False)
news_list = []

print("Loading NLP Models...")
finbert_base = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0, batch_size=32, truncation=True)
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", device=0, batch_size=32, truncation=True)
models = [finbert_base, finbert_tone]
stocks = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "NFLX"]


print("Fetching News")
try:
    for stock in stocks:
        news = client.list_ticker_news(stock, limit=100)
        for item in news:
            news_list.append({
                "headline": item.title,
                # Force to UTC to ensure it matches yfinance Datetime
                "date": pd.to_datetime(item.published_utc, utc=True), 
                "ticker": stock
            })
        time.sleep(15)
except Exception as e:
    print(f"News fetch error: {e}")

def normalize(label, score):
    label = label.lower()
    if label == 'positive': return score
    elif label == 'negative': return -score
    else: return 0

def combine_results(headlines):
    scores = []
    for model in models:
        results = model(headlines)
        score = [normalize(result['label'], result['score']) for result in results]
        scores.append(score)
    return np.mean(np.array(scores), axis=0)

print("Scoring")
headline_texts = [item["headline"] for item in news_list]



scores = combine_results(headline_texts)


print("Merging")
final = []
for item, score in zip(news_list, scores):
    ts = item["date"]
    current_price = get_price_date(item["ticker"], ts)
    future_price = get_price_change(item["ticker"], ts)                
    
    if current_price is None or future_price is None:
        continue

    new_row = {
        "headline": item["headline"],
        "date": item["date"],
        "ticker": item["ticker"],
        "score": float(score),
        "current": current_price["Close"],
        "1h_return": future_price
    }

    for col in current_price.index:                
        if col not in ["Datetime", "Open", "High", "Low", "Close", "Volume"]:
            new_row[col] = current_price[col] 

    final.append(new_row)


df = pd.DataFrame(final)

df = df.dropna() 

df.to_csv("data.csv", index=False)
print("saved")