# Quick Start Guide - Streamlit App

## 🎯 Run the App Locally

```bash
# From project root
./run_app.sh

# Or manually
streamlit run app.py
```

**App will open at:** http://localhost:8501

## 📖 Step-by-Step Usage

### Step 1: Load Data (Page 1)
1. Enter a ticker symbol (e.g., **SPY**, **AAPL**, **BTC-USD**)
2. Select date range (default: last 2 years)
3. Click **"Load Data"**
4. View price charts, volume, and statistics
5. Generate technical features (50+ indicators)

**Example tickers to try:**
- `SPY` - S&P 500 ETF
- `AAPL` - Apple stock
- `BTC-USD` - Bitcoin
- `QQQ` - NASDAQ ETF
- `TLT` - Treasury bonds

### Step 2: Detect Regimes (Page 2)
1. Choose detection model:
   - **GMM**: Fast, good for clear regimes
   - **HMM**: Better for temporal patterns
2. Select number of regimes (2-5)
   - 2: Bull/Bear
   - 3: Bull/Bear/Sideways (recommended)
   - 4+: More granular
3. Label regimes (optional)
4. Click **"Detect Regimes"**
5. View:
   - Price chart with colored regime overlays
   - Regime statistics (returns, volatility)
   - Transition probabilities

### Step 3: Analyze Strategies (Page 3)
1. Select strategies to test:
   - ✅ Trend Following
   - ✅ Mean Reversion
   - ✅ Volatility Breakout
   - ✅ Regime-Adaptive
2. Configure parameters (optional)
3. Click **"Run Analysis"**
4. View:
   - Equity curves
   - Performance metrics (Sharpe, Sortino, Max DD)
   - Best strategy per regime
   - Risk analysis

### Step 4: Live Dashboard (Page 4)
- View current market status
- Current regime and confidence
- Technical signals (RSI, MACD, Bollinger Bands)
- AI-generated action items
- Recent regime transitions

## 💡 Example Workflows

### Workflow 1: Find Bull/Bear Markets
```
1. Load SPY (2020-01-01 to present)
2. Detect 2 regimes with GMM
3. Label: "Bull" and "Bear"
4. Analyze Trend Following vs Mean Reversion
5. Result: Bull markets favor Trend Following
```

### Workflow 2: Crypto Volatility Trading
```
1. Load BTC-USD (last 2 years)
2. Detect 3 regimes with HMM
3. Label: "High Vol", "Low Vol", "Trending"
4. Test Volatility Breakout strategy
5. Result: High Vol regimes best for breakouts
```

### Workflow 3: Stock Mean Reversion
```
1. Load AAPL (3 years)
2. Detect 3 regimes
3. Analyze Mean Reversion vs Buy & Hold
4. Result: Sideways regimes favor Mean Reversion
```

## 🚀 Deploy to Cloud (FREE)

### Option 1: Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect GitHub and select repository
4. Deploy (takes 2 minutes)
5. Share public URL with anyone

### Option 2: Heroku
```bash
# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku main

# Open
heroku open
```

## 🎨 Features Overview

| Feature | Description | Page |
|---------|-------------|------|
| Real-time Data | Yahoo Finance API (no key needed) | 1 |
| Technical Indicators | 50+ features (RSI, MACD, BB, etc.) | 1 |
| Regime Detection | GMM, HMM, DTW clustering | 2 |
| Strategy Backtesting | Multiple strategies with metrics | 3 |
| Live Insights | Current market intelligence | 4 |
| Interactive Charts | Plotly visualizations | All |

## ⚡ Performance Tips

- **Faster loading**: Use cached data
- **Faster regimes**: Use GMM, reduce date range
- **Better accuracy**: Use HMM with more data
- **Compare strategies**: Test 3-4 simultaneously

## 🐛 Common Issues

**Issue**: "No data retrieved"
- **Fix**: Check ticker symbol spelling, try different date range

**Issue**: "Regime detection slow"
- **Fix**: Reduce date range or use GMM instead of HMM

**Issue**: "Strategy returns NaN"
- **Fix**: Ensure regime detection completed successfully first

## 📊 Understanding Results

### Regime Statistics
- **Mean Return**: Average daily return in this regime
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted returns (>1 is good)
- **Frequency**: % of time in this regime
- **Avg Duration**: How long regime persists

### Strategy Metrics
- **Total Return**: Cumulative gain/loss
- **Annualized Return**: Yearly equivalent return
- **Sharpe Ratio**: Return per unit of risk (>1 good, >2 excellent)
- **Sortino Ratio**: Like Sharpe but only downside risk
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: % of profitable periods

### Technical Signals
- **RSI > 70**: Overbought (potential sell)
- **RSI < 30**: Oversold (potential buy)
- **MACD Cross**: Momentum change
- **BB Breakout**: Volatility expansion

## 🎯 Next Steps

1. **Experiment**: Try different tickers and date ranges
2. **Customize**: Modify strategy parameters
3. **Compare**: Test multiple models (GMM vs HMM)
4. **Deploy**: Share with others on Streamlit Cloud
5. **Enhance**: Add your own strategies or indicators

## 📚 Resources

- **Full Docs**: README_STREAMLIT.md
- **Project Overview**: README.md
- **Code Guide**: CLAUDE.md
- **Streamlit Docs**: https://docs.streamlit.io

---

**🎉 You're ready to go! Start with SPY and 3 regimes to see it in action.**