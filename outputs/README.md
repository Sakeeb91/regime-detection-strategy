# Visualizations & Simulation Outputs

This folder contains all generated plots and simulation data from the Market Regime Detection system.

## 📁 Folder Structure

```
outputs/
├── plots/              # Professional visualizations (300 DPI)
│   ├── 01_raw_data.png
│   ├── 02_returns_analysis.png
│   ├── 03_technical_indicators.png
│   ├── 04_gmm_regimes.png
│   └── ...
├── simulations/        # Simulation data exports
│   ├── simulation_SPY_*.json
│   └── returns_SPY_*.csv
└── screenshots/        # Streamlit app screenshots (optional)
```

## 📊 Available Plots

### Data & Preprocessing
- **01_raw_data.png**: OHLCV price and volume data
- **02_returns_analysis.png**: Returns distribution, cumulative performance, volatility
- **03_technical_indicators.png**: RSI, MACD, Bollinger Bands, ATR

### Regime Detection
- **04_gmm_regimes.png**: GMM regime overlay on price chart
- **05_gmm_statistics.png**: Performance metrics by regime
- **06_gmm_probabilities.png**: Regime probability evolution
- **07_hmm_regimes.png**: HMM regime overlay
- **08_gmm_vs_hmm.png**: Model comparison

### Strategy Performance
- **09_equity_curves.png**: Strategy returns comparison
- **10_performance_metrics.png**: Sharpe, returns, drawdown analysis
- **11_drawdown_analysis.png**: Drawdown charts
- **12_regime_adaptive.png**: Adaptive strategy performance
- **13_executive_summary.png**: Comprehensive results summary

## 🔄 How to Generate

Run the demo script to generate all visualizations:

```bash
python demo_with_plots.py
```

This will:
1. Load SPY data from Yahoo Finance
2. Run complete ML pipeline
3. Generate all 13 plots
4. Save simulation data
5. Create index file

## 📈 For Portfolio/Presentations

**Recommended order for showcasing:**
1. Start with **13_executive_summary.png** (overview)
2. Show **04_gmm_regimes.png** (ML capability)
3. Show **09_equity_curves.png** (results)
4. Show **12_regime_adaptive.png** (innovation)

**For recruiters:**
- All plots are print-quality (300 DPI)
- Each has description file explaining the visualization
- Can be used in presentations, portfolios, papers

## 📝 Plot Descriptions

Each plot has an accompanying `*_description.txt` file that explains:
- What the visualization shows
- How to interpret it
- Key insights and patterns
- Technical context

## 💾 Simulation Data

**JSON Format** (`simulations/*.json`):
- Complete simulation results
- Regime statistics
- Strategy metrics
- Configuration parameters
- Timestamps

**CSV Format** (`simulations/*.csv`):
- Daily returns for all strategies
- Regime labels
- Market returns
- Easy to import into Excel/pandas

## 🎯 Use Cases

### Academic/Research
- Include in papers or presentations
- Demonstrate methodology
- Show empirical results

### Portfolio/Resume
- Visual proof of ML capabilities
- End-to-end project demonstration
- Professional presentation quality

### Business/Stakeholders
- Executive summaries
- Performance comparisons
- Risk analysis

## 🔧 Customization

To generate plots for different tickers or date ranges:

```python
from demo_with_plots import run_complete_simulation

# Custom ticker and dates
run_complete_simulation('AAPL', start_date='2021-01-01')
run_complete_simulation('BTC-USD', start_date='2020-01-01')
```

## 📊 Interactive Versions

For interactive versions of these visualizations, use the Streamlit app:

```bash
streamlit run app.py
```

Interactive features include:
- Zoom and pan
- Hover for details
- Dynamic parameter adjustment
- Real-time updates

---

**Generated:** 2025-09-30
**Status:** 4 plots available (more coming with fixed demo script)
**Quality:** 300 DPI, professional color scheme
**Format:** PNG with accompanying descriptions