"""
Utility functions to fetch and display company information.
"""

import yfinance as yf
from typing import Dict, Optional
from loguru import logger


def get_company_info(ticker: str) -> Dict[str, str]:
    """
    Fetch company information for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'SPY')

    Returns:
        Dictionary with company information:
        - name: Company name
        - sector: GICS sector
        - industry: GICS industry
        - country: Headquarters country
        - city: Headquarters city
        - founded: Year founded (if available)
        - website: Company website
        - description: Brief description

    Examples:
        >>> info = get_company_info('AAPL')
        >>> print(info['name'])
        'Apple Inc.'
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract relevant information with fallbacks
        company_data = {
            'name': info.get('longName') or info.get('shortName') or ticker,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'city': info.get('city', 'N/A'),
            'website': info.get('website', 'N/A'),
            'description': info.get('longBusinessSummary', 'No description available'),
            'employees': info.get('fullTimeEmployees', 'N/A'),
            'market_cap': info.get('marketCap', None),
        }

        # Try to get founding year from various sources
        # Note: yfinance doesn't always provide founding year directly
        # We'll show N/A if not available
        company_data['founded'] = 'N/A'

        logger.info(f"Successfully fetched company info for {ticker}")
        return company_data

    except Exception as e:
        logger.warning(f"Could not fetch company info for {ticker}: {str(e)}")

        # Return minimal info on error
        return {
            'name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'country': 'N/A',
            'city': 'N/A',
            'website': 'N/A',
            'description': 'Information not available',
            'employees': 'N/A',
            'market_cap': None,
            'founded': 'N/A',
        }


def format_market_cap(market_cap: Optional[int]) -> str:
    """
    Format market cap in human-readable format.

    Args:
        market_cap: Market capitalization in dollars

    Returns:
        Formatted string (e.g., '$2.5T', '$150B', '$5.2M')

    Examples:
        >>> format_market_cap(2500000000000)
        '$2.5T'
    """
    if market_cap is None or market_cap == 'N/A':
        return 'N/A'

    try:
        market_cap = float(market_cap)

        if market_cap >= 1_000_000_000_000:  # Trillion
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:  # Billion
            return f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:  # Million
            return f"${market_cap / 1_000_000:.2f}M"
        else:
            return f"${market_cap:,.0f}"
    except (ValueError, TypeError):
        return 'N/A'


def format_employees(employees) -> str:
    """
    Format employee count with thousands separator.

    Args:
        employees: Number of employees

    Returns:
        Formatted string (e.g., '154,000')
    """
    if employees is None or employees == 'N/A':
        return 'N/A'

    try:
        return f"{int(employees):,}"
    except (ValueError, TypeError):
        return 'N/A'