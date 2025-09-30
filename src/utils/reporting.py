"""
Reporting utilities for generating performance reports.

Creates comprehensive HTML and text reports with strategy performance,
regime analysis, and risk metrics.
"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


class PerformanceReporter:
    """
    Generate comprehensive performance reports.

    Creates detailed reports including strategy performance, regime analysis,
    trade statistics, and risk metrics.

    Examples:
        >>> reporter = PerformanceReporter()
        >>> report = reporter.generate_report(backtest_results)
        >>> reporter.save_html_report(report, 'reports/strategy_report.html')
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize reporter.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PerformanceReporter initialized: output_dir={output_dir}")

    def generate_report(
        self, backtest_results: Dict, strategy_name: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive performance report.

        Args:
            backtest_results: Results from Backtester.run()
            strategy_name: Optional strategy name override

        Returns:
            Dictionary with report sections
        """
        if strategy_name is None:
            strategy_name = backtest_results.get("strategy_name", "Unknown Strategy")

        report = {
            "metadata": self._generate_metadata(strategy_name),
            "executive_summary": self._generate_executive_summary(backtest_results),
            "performance_metrics": self._generate_performance_section(backtest_results),
            "trade_analysis": self._generate_trade_analysis(backtest_results),
            "regime_analysis": self._generate_regime_section(backtest_results),
            "risk_metrics": self._generate_risk_section(backtest_results),
        }

        return report

    def _generate_metadata(self, strategy_name: str) -> Dict:
        """Generate report metadata."""
        return {
            "strategy_name": strategy_name,
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generator": "Market Regime Detection System",
        }

    def _generate_executive_summary(self, results: Dict) -> Dict:
        """Generate executive summary section."""
        metrics = results["metrics"]

        summary = {
            "total_return": metrics["total_return"],
            "cagr": metrics["cagr"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "total_trades": metrics["n_trades"],
        }

        # Performance vs buy & hold
        if "buy_hold_return" in metrics:
            summary["outperformance"] = (
                metrics["total_return"] - metrics["buy_hold_return"]
            )

        return summary

    def _generate_performance_section(self, results: Dict) -> Dict:
        """Generate detailed performance metrics section."""
        metrics = results["metrics"]

        performance = {
            "returns": {
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "buy_hold_return": metrics.get("buy_hold_return", None),
            },
            "risk_adjusted": {
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "calmar_ratio": metrics["calmar_ratio"],
            },
            "volatility": {"annual_volatility": metrics["annual_volatility"]},
            "drawdown": {"max_drawdown": metrics["max_drawdown"]},
        }

        return performance

    def _generate_trade_analysis(self, results: Dict) -> Dict:
        """Generate trade analysis section."""
        trades_df = results["trades"]
        metrics = results["metrics"]

        if len(trades_df) == 0:
            return {"total_trades": 0, "message": "No trades executed"}

        # Calculate trade statistics
        winning_trades = trades_df[trades_df["return"] > 0]
        losing_trades = trades_df[trades_df["return"] < 0]

        analysis = {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "avg_win": (
                winning_trades["return"].mean() if len(winning_trades) > 0 else 0
            ),
            "avg_loss": losing_trades["return"].mean() if len(losing_trades) > 0 else 0,
            "largest_win": trades_df["return"].max(),
            "largest_loss": trades_df["return"].min(),
            "avg_holding_period": trades_df["holding_period"].mean(),
            "long_trades": len(trades_df[trades_df["direction"] == "Long"]),
            "short_trades": len(trades_df[trades_df["direction"] == "Short"]),
        }

        return analysis

    def _generate_regime_section(self, results: Dict) -> Optional[Dict]:
        """Generate regime-specific analysis section."""
        metrics = results["metrics"]

        if "regime_analysis" not in metrics:
            return None

        regime_analysis = metrics["regime_analysis"]

        regime_summary = {}
        for regime_key, regime_metrics in regime_analysis.items():
            regime_summary[regime_key] = {
                "periods": regime_metrics["n_periods"],
                "total_return": regime_metrics["total_return"],
                "sharpe": regime_metrics["sharpe"],
                "trades": regime_metrics["n_trades"],
                "win_rate": regime_metrics["win_rate"],
            }

        return regime_summary

    def _generate_risk_section(self, results: Dict) -> Dict:
        """Generate risk metrics section."""
        results["equity_curve"]
        returns = results["returns"]

        # Calculate additional risk metrics
        var_95 = np.percentile(returns, 5)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean()  # 95% CVaR

        # Calculate consecutive losses
        losing_streak = self._calculate_max_streak(returns < 0)
        winning_streak = self._calculate_max_streak(returns > 0)

        risk_metrics = {
            "value_at_risk_95": var_95,
            "conditional_var_95": cvar_95,
            "max_consecutive_losses": losing_streak,
            "max_consecutive_wins": winning_streak,
            "downside_volatility": returns[returns < 0].std() * np.sqrt(252),
        }

        return risk_metrics

    def _calculate_max_streak(self, condition_series: pd.Series) -> int:
        """Calculate maximum consecutive streak."""
        streaks = (
            condition_series.astype(int)
            .groupby((condition_series != condition_series.shift()).cumsum())
            .sum()
        )
        return int(streaks.max()) if len(streaks) > 0 else 0

    def generate_text_report(self, report: Dict) -> str:
        """
        Generate formatted text report.

        Args:
            report: Report dictionary from generate_report()

        Returns:
            Formatted text report
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(f"PERFORMANCE REPORT: {report['metadata']['strategy_name']}")
        lines.append(f"Generated: {report['metadata']['report_date']}")
        lines.append("=" * 80)
        lines.append("")

        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        summary = report["executive_summary"]
        lines.append(f"Total Return:        {summary['total_return']:>10.2%}")
        lines.append(f"CAGR:                {summary['cagr']:>10.2%}")
        lines.append(f"Sharpe Ratio:        {summary['sharpe_ratio']:>10.2f}")
        lines.append(f"Max Drawdown:        {summary['max_drawdown']:>10.2%}")
        lines.append(f"Win Rate:            {summary['win_rate']:>10.2%}")
        lines.append(f"Total Trades:        {summary['total_trades']:>10d}")

        if "outperformance" in summary:
            lines.append(f"Outperformance:      {summary['outperformance']:>10.2%}")
        lines.append("")

        # Performance Metrics
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 80)
        perf = report["performance_metrics"]
        lines.append(
            f"Sortino Ratio:       {perf['risk_adjusted']['sortino_ratio']:>10.2f}"
        )
        lines.append(
            f"Calmar Ratio:        {perf['risk_adjusted']['calmar_ratio']:>10.2f}"
        )
        lines.append(
            f"Annual Volatility:   {perf['volatility']['annual_volatility']:>10.2%}"
        )
        lines.append("")

        # Trade Analysis
        lines.append("TRADE ANALYSIS")
        lines.append("-" * 80)
        trade = report["trade_analysis"]
        if trade["total_trades"] > 0:
            lines.append(f"Winning Trades:      {trade['winning_trades']:>10d}")
            lines.append(f"Losing Trades:       {trade['losing_trades']:>10d}")
            lines.append(f"Profit Factor:       {trade['profit_factor']:>10.2f}")
            lines.append(f"Avg Win:             {trade['avg_win']:>10.2%}")
            lines.append(f"Avg Loss:            {trade['avg_loss']:>10.2%}")
            lines.append(f"Largest Win:         {trade['largest_win']:>10.2%}")
            lines.append(f"Largest Loss:        {trade['largest_loss']:>10.2%}")
            lines.append(
                f"Avg Hold Period:     {trade['avg_holding_period']:>10.1f} days"
            )
            lines.append(f"Long Trades:         {trade['long_trades']:>10d}")
            lines.append(f"Short Trades:        {trade['short_trades']:>10d}")
        else:
            lines.append("No trades executed")
        lines.append("")

        # Risk Metrics
        lines.append("RISK METRICS")
        lines.append("-" * 80)
        risk = report["risk_metrics"]
        lines.append(f"Value at Risk (95%): {risk['value_at_risk_95']:>10.2%}")
        lines.append(f"CVaR (95%):          {risk['conditional_var_95']:>10.2%}")
        lines.append(f"Max Losing Streak:   {risk['max_consecutive_losses']:>10d}")
        lines.append(f"Max Winning Streak:  {risk['max_consecutive_wins']:>10d}")
        lines.append(f"Downside Vol:        {risk['downside_volatility']:>10.2%}")
        lines.append("")

        # Regime Analysis
        if report["regime_analysis"] is not None:
            lines.append("REGIME ANALYSIS")
            lines.append("-" * 80)
            for regime_key, regime_data in report["regime_analysis"].items():
                lines.append(f"\n{regime_key.upper()}:")
                lines.append(f"  Periods:           {regime_data['periods']:>10d}")
                lines.append(
                    f"  Total Return:      {regime_data['total_return']:>10.2%}"
                )
                lines.append(f"  Sharpe:            {regime_data['sharpe']:>10.2f}")
                lines.append(f"  Trades:            {regime_data['trades']:>10d}")
                lines.append(f"  Win Rate:          {regime_data['win_rate']:>10.2%}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save_text_report(self, report: Dict, filename: str):
        """
        Save report as text file.

        Args:
            report: Report dictionary
            filename: Output filename
        """
        text_report = self.generate_text_report(report)
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(text_report)

        logger.info(f"Text report saved to {filepath}")

    def save_html_report(self, report: Dict, filename: str):
        """
        Save report as HTML file.

        Args:
            report: Report dictionary
            filename: Output filename
        """
        html = self._generate_html(report)
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(html)

        logger.info(f"HTML report saved to {filepath}")

    def _generate_html(self, report: Dict) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report['metadata']['strategy_name']} - Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px;
                background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white;
                      padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd;
              padding-bottom: 5px; }}
        .metric {{ display: inline-block; width: 30%; margin: 10px 1%;
                   padding: 15px; background-color: #f9f9f9;
                   border-left: 4px solid #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #777; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report['metadata']['strategy_name']}</h1>
        <p><em>Generated: {report['metadata']['report_date']}</em></p>

        <h2>Executive Summary</h2>
        <div class="metrics">
"""

        # Add executive summary metrics
        summary = report["executive_summary"]
        for key, value in summary.items():
            if isinstance(value, float):
                if "return" in key or "performance" in key:
                    value_str = f"{value:.2%}"
                    css_class = "positive" if value > 0 else "negative"
                elif "ratio" in key:
                    value_str = f"{value:.2f}"
                    css_class = "positive" if value > 0 else "negative"
                else:
                    value_str = f"{value:.2%}" if abs(value) < 1 else f"{value:.2f}"
                    css_class = "positive" if value > 0 else "negative"
            else:
                value_str = str(value)
                css_class = ""

            label = key.replace("_", " ").title()
            html += f"""
            <div class="metric">
                <div class="metric-label">{label}</div>
                <div class="metric-value {css_class}">{value_str}</div>
            </div>
"""

        html += """
        </div>

        <h2>Trade Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""

        # Add trade analysis
        trade = report["trade_analysis"]
        if trade["total_trades"] > 0:
            trade_metrics = [
                ("Total Trades", trade["total_trades"]),
                ("Winning Trades", trade["winning_trades"]),
                ("Losing Trades", trade["losing_trades"]),
                ("Win Rate", f"{trade['win_rate']:.2%}"),
                ("Profit Factor", f"{trade['profit_factor']:.2f}"),
                ("Average Win", f"{trade['avg_win']:.2%}"),
                ("Average Loss", f"{trade['avg_loss']:.2%}"),
                ("Largest Win", f"{trade['largest_win']:.2%}"),
                ("Largest Loss", f"{trade['largest_loss']:.2%}"),
            ]

            for metric, value in trade_metrics:
                html += f"<tr><td>{metric}</td><td>{value}</td></tr>\n"

        html += """
        </table>

        <h2>Risk Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""

        # Add risk metrics
        risk = report["risk_metrics"]
        risk_metrics = [
            ("Value at Risk (95%)", f"{risk['value_at_risk_95']:.2%}"),
            ("Conditional VaR (95%)", f"{risk['conditional_var_95']:.2%}"),
            ("Max Losing Streak", risk["max_consecutive_losses"]),
            ("Max Winning Streak", risk["max_consecutive_wins"]),
            ("Downside Volatility", f"{risk['downside_volatility']:.2%}"),
        ]

        for metric, value in risk_metrics:
            html += f"<tr><td>{metric}</td><td>{value}</td></tr>\n"

        html += """
        </table>
    </div>
</body>
</html>
"""

        return html

    def compare_strategies(
        self,
        strategy_results: Dict[str, Dict],
        filename: str = "strategy_comparison.html",
    ):
        """
        Generate comparison report for multiple strategies.

        Args:
            strategy_results: Dict mapping strategy names to backtest results
            filename: Output filename
        """
        comparison_data = []

        for strategy_name, results in strategy_results.items():
            metrics = results["metrics"]
            comparison_data.append(
                {
                    "Strategy": strategy_name,
                    "Total Return": metrics["total_return"],
                    "CAGR": metrics["cagr"],
                    "Sharpe": metrics["sharpe_ratio"],
                    "Sortino": metrics["sortino_ratio"],
                    "Max DD": metrics["max_drawdown"],
                    "Win Rate": metrics["win_rate"],
                    "Trades": metrics["n_trades"],
                }
            )

        df = pd.DataFrame(comparison_data)

        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .best {{ background-color: #c8e6c9; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Strategy Comparison Report</h1>
        <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        {df.to_html(index=False, classes='table')}
    </div>
</body>
</html>
"""

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(html)

        logger.info(f"Comparison report saved to {filepath}")
