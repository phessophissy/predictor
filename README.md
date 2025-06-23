[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/phessophissy/predictor)

# Crypto Price Predictor

A minimal web AI app to predict the next price of a cryptocurrency using recent price data and a simple AI model.

## Setup

1. **Install dependencies:**

```bash
# (Optional) Activate your virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run the backend:**

```bash
uvicorn main:app --reload
```

3. **Use the frontend:**

Open `index.html` in your browser. Enter a crypto symbol (e.g., `bitcoin`, `ethereum`) and click Predict.

- The backend must be running at `http://localhost:8000`.
- The frontend will call the `/predict` endpoint and display the predicted and last price.

## Notes
- Uses CoinGecko API for recent price data.
- The AI model is a simple linear regression on the last 24 hours of hourly prices. 