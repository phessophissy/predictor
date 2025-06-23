from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory if needed in the future
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionRequest(BaseModel):
    symbol: str  # e.g., 'bitcoin', 'ethereum'

class PredictionResponse(BaseModel):
    predicted_price: float
    last_price: float
    symbol: str

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"

@app.post("/predict", response_model=PredictionResponse)
def predict_price(req: PredictionRequest):
    # Fetch last 24 hours of hourly prices
    url = COINGECKO_URL.format(symbol=req.symbol)
    params = {"vs_currency": "usd", "days": 1, "interval": "hourly"}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return {"predicted_price": None, "last_price": None, "symbol": req.symbol}
    data = resp.json()
    prices = data.get("prices", [])
    if len(prices) < 2:
        return {"predicted_price": None, "last_price": None, "symbol": req.symbol}
    # Prepare data for regression
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array([p[1] for p in prices])
    model = LinearRegression()
    model.fit(X, y)
    next_time = np.array([[len(prices)]])
    predicted_price = float(model.predict(next_time)[0])
    last_price = float(y[-1])
    return PredictionResponse(predicted_price=predicted_price, last_price=last_price, symbol=req.symbol)

@app.get("/")
def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html")) 