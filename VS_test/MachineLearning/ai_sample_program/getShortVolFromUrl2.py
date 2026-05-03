import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta

# Finnhub API key (free signup at finnhub.io)
FINNHUB_API_KEY = "your_finnhub_api_key_here"  # Replace with your key (optional for yfinance only)


def get_short_interest(symbol, target_date=None, use_finnhub=True):
    """
    Fetch short interest for a symbol on or near the target date.

    Args:
        symbol (str): e.g., 'NVDA'
        target_date (str): e.g., '2025-12-03' (YYYY-MM-DD)
        use_finnhub (bool): Use Finnhub API if key provided

    Returns:
        dict: Short interest data or None
    """
    if target_date:
        try:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format. Use YYYY-MM-DD.")
            return None

    # Method 1: yfinance (approximate, latest available)
    ticker = yf.Ticker(symbol)
    info = ticker.info
    yf_data = {
        'symbol': symbol,
        'short_percent_of_float': info.get('shortPercentOfFloat', 'N/A'),
        'shares_short': info.get('sharesShort', 'N/A'),
        'report_date': info.get('lastFiscalYearEnd', 'N/A'),  # Approximate
        'source': 'yfinance'
    }

    # Method 2: Finnhub (historical, if key provided)
    if use_finnhub and FINNHUB_API_KEY != "your_finnhub_api_key_here":
        url = f"https://finnhub.io/api/v1/stock/short-interest?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data and len(data.get('results', [])) > 0:
                latest = data['results'][0]
                fh_data = {
                    'symbol': symbol,
                    'short_interest_shares': latest.get('totalSharesShort', 'N/A'),
                    'short_percent_float': latest.get('shortPercentOfFloat', 'N/A'),
                    'days_to_cover': latest.get('daysToCover', 'N/A'),
                    'report_date': latest.get('reportDateTime', 'N/A'),
                    'source': 'finnhub'
                }
                # Filter closest to target_date if provided
                if target_date:
                    # Find closest report (simplified; enhance with date parsing if needed)
                    fh_data['closest_to_target'] = abs(
                        datetime.strptime(fh_data['report_date'], '%Y-%m-%d') - target_dt).days
                return fh_data

    # Return yfinance data as fallback
    return yf_data


# ==============================================================
# Usage
# ==============================================================
if __name__ == "__main__":
    symbols = ['NVDA', 'TSLA']
    target_date = '2025-12-03'  # Your specific date

    results = []
    for symbol in symbols:
        data = get_short_interest(symbol, target_date, use_finnhub=True)
        if data:
            results.append(data)
            print(f"\n{symbol} Short Interest (closest to {target_date}):")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"No data for {symbol}")

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv('short_interest_2025-12-03.csv', index=False)
        print(f"\nSaved to 'short_interest_2025-12-03.csv'")
        print(df)