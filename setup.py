"""Setup configuration for regime-detection-strategy package."""

from setuptools import setup, find_packages

setup(
    name="regime-detection-strategy",
    version="0.1.0",
    description="Market Regime Detection & Adaptive Strategy Selection",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "arch>=6.0.0",
        "yfinance>=0.2.28",
        "xgboost>=2.0.0",
        "hmmlearn>=0.3.0",
        "tslearn>=0.6.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "pylint>=2.17.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ]
    },
)