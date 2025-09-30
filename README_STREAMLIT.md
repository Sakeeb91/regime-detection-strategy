# Market Regime Detection & Adaptive Trading - Streamlit App

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
./run_app.sh
# OR
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📱 Features

### 1. Data Explorer
- **Real-time data fetching** from Yahoo Finance (no API key required)
- **Interactive charts**: Candlestick, volume, returns distribution
- **Technical indicators**: 50+ features across 5 categories
- **Supported assets**: Stocks, ETFs, Crypto, Indices

### 2. Regime Detection
- **ML Models**: GMM, HMM, DTW clustering
- **Visual regime overlay** on price charts
- **Regime statistics**: Returns, volatility, duration by regime
- **Transition analysis**: Probability matrices and patterns

### 3. Strategy Analysis
- **Multiple strategies**: Trend Following, Mean Reversion, Volatility Breakout
- **Regime-adaptive** strategy selection
- **Performance metrics**: Sharpe, Sortino, Max Drawdown
- **Backtesting**: Transaction costs, slippage modeling

### 4. Live Dashboard
- **Current market status**: Price, volatility, momentum
- **Regime intelligence**: Current regime, confidence, transition alerts
- **Technical signals**: MA, RSI, MACD, Bollinger Bands
- **Action items**: AI-generated recommendations

## 🎯 Usage Workflow

1. **Load Data**: Select ticker (e.g., SPY, AAPL) and date range
2. **Detect Regimes**: Choose model (GMM/HMM) and number of regimes
3. **Analyze Strategies**: Compare performance across regimes
4. **Monitor Live**: Get real-time insights and recommendations

## 📊 Example Use Cases

### Use Case 1: S&P 500 Regime Analysis
```
Ticker: SPY
Date Range: 2020-01-01 to Present
Model: GMM with 3 regimes
Result: Identify Bull/Bear/Sideways markets + optimal strategies
```

### Use Case 2: Crypto Volatility Trading
```
Ticker: BTC-USD
Date Range: Last 2 years
Model: HMM with 4 regimes
Result: High/Low volatility regime detection for breakout trading
```

### Use Case 3: Tech Stock Momentum
```
Ticker: AAPL, TSLA
Date Range: Last 3 years
Strategies: Compare Trend Following vs Mean Reversion
Result: Regime-based strategy selection
```

## 🌐 Deployment

### Deploy to Streamlit Cloud (FREE)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "feat: add Streamlit app for regime detection"
   git push
   ```

2. **Deploy on Streamlit Cloud**:
   - Visit: https://share.streamlit.io
   - Click "New app"
   - Select your GitHub repository
   - Main file: `app.py`
   - Click "Deploy"

### Deploy to Heroku

1. **Create Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.headless=true
   ```

2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Deploy to AWS/GCP/Azure

Use Docker:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🎨 Customization

### Modify Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#4CAF50"
backgroundColor = "#0E1117"
```

### Add New Strategies
1. Create strategy class in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Add to `pages/3_Strategy_Analysis.py`

### Add New Indicators
1. Update `src/data/feature_engineer.py`
2. Add to Data Explorer visualization

## 📈 Performance Tips

- **Cache data**: App automatically caches downloaded data
- **Reduce date range**: Shorter ranges = faster processing
- **Limit features**: Use regime-specific features for faster detection
- **Batch analysis**: Analyze multiple tickers separately

## 🐛 Troubleshooting

### Issue: Data not loading
- **Solution**: Check internet connection, try different ticker symbol

### Issue: Regime detection slow
- **Solution**: Reduce date range or use GMM instead of HMM

### Issue: Charts not rendering
- **Solution**: Update browser, clear cache, refresh page

## 📝 Data Sources

- **Yahoo Finance** (yfinance): Free, no API key required
- **Supported**: Stocks, ETFs, Indices, Crypto, Forex
- **Frequency**: Daily (can be extended to intraday)

## 🔐 Security Notes

- No API keys stored in code
- All data fetched in real-time
- No user data collected
- Run locally for sensitive analysis

## 📚 Documentation

- **Project Docs**: See main README.md
- **Streamlit Docs**: https://docs.streamlit.io
- **API Docs**: See CLAUDE.md

## 🤝 Contributing

Improvements welcome! Focus areas:
- New regime detection models
- Additional trading strategies
- Enhanced visualizations
- Real-time data streaming

## 📄 License

MIT License - See LICENSE file

---

**Built with:** Python • Streamlit • Plotly • scikit-learn • yfinance

**Not financial advice** - For educational and research purposes only.