import pandas as pd
import numpy as np
import yfinance as yf
import gzip
import os
from datetime import datetime, timedelta
from fredapi import Fred
import re
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# S&P 100 tickers (major components)
# For faster testing, you can uncomment the line below to use only a subset
# SP100_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'MA']

SP100_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'V', 'UNH',
    'XOM', 'JNJ', 'JPM', 'WMT', 'MA', 'PG', 'CVX', 'LLY', 'HD', 'MRK',
    'ABBV', 'AVGO', 'KO', 'PEP', 'COST', 'ADBE', 'MCD', 'CRM', 'CSCO', 'TMO',
    'ACN', 'NFLX', 'ABT', 'DHR', 'INTC', 'AMD', 'WFC', 'DIS', 'VZ', 'CMCSA',
    'PFE', 'TXN', 'NKE', 'UPS', 'PM', 'ORCL', 'NEE', 'COP', 'RTX', 'HON',
    'QCOM', 'LOW', 'BA', 'SBUX', 'IBM', 'CAT', 'GE', 'GS', 'AXP',
    'AMGN', 'BLK', 'SPGI', 'ISRG', 'DE', 'BKNG', 'MMM', 'TJX', 'CVS', 'ELV',
    'MDLZ', 'CI', 'ADP', 'GILD', 'SYK', 'ZTS', 'BMY', 'ADI', 'REGN', 'VRTX',
    'AMT', 'PLD', 'SO', 'DUK', 'MO', 'USB', 'TFC', 'PNC', 'SCHW', 'MS',
    'C', 'BK', 'AIG', 'MET', 'CB', 'TRV', 'AON', 'MMC', 'ICE', 'CME'
]

# Date range for data retrieval (December 2025 only for faster processing)
START_DATE = datetime(2025, 12, 1)
END_DATE = datetime(2025, 12, 31)

# FRED API key (you'll need to get your own from https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = '4a80f3bf51df1985e6569f15a90c890e'  # Replace with your actual key

# Path to local options data
OPTIONS_DATA_PATH = r"C:\Users\diefr\OneDrive\Lernsachen\MarketConsensus\optiondata"

# Directories for data storage
DATA_DIR = './data'
STOCK_DATA_DIR = os.path.join(DATA_DIR, 'stocks')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

for directory in [DATA_DIR, STOCK_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_option_ticker(ticker):
    """
    Parse option ticker to extract components.
    Format: O:AAPL250117C00150000 (with O: prefix from massive.com)
    - O: prefix for options
    - AAPL: underlying symbol (variable length)
    - 250117: expiration date YYMMDD
    - C/P: call or put
    - 00150000: strike price (8 digits, divide by 1000)
    """
    try:
        # Remove O: prefix if present
        if ticker.startswith('O:'):
            ticker = ticker[2:]
        
        # Match pattern: letters (underlying) + 6 digits (date) + C/P + 8 digits (strike)
        match = re.match(r'^([A-Z\.]+)(\d{6})([CP])(\d{8})$', ticker)
        if not match:
            return None
        
        underlying, date_str, option_type, strike_str = match.groups()
        
        # Parse expiration date
        expiration = datetime.strptime(date_str, '%y%m%d')
        
        # Parse strike price (divide by 1000 to get actual strike)
        strike = float(strike_str) / 1000
        
        return {
            'underlying': underlying,
            'expiration': expiration,
            'option_type': option_type,
            'strike': strike
        }
    except Exception as e:
        return None

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def implied_volatility_fast(option_price, S, K, T, r, option_type='C'):
    """
    Calculate implied volatility using optimized Brent's method with better initial guess
    Uses Brenner-Subrahmanyam approximation for initial guess
    """
    if T <= 0 or option_price <= 0 or S <= 0 or K <= 0:
        return np.nan
    
    # Intrinsic value
    if option_type == 'C':
        intrinsic = max(S - K, 0)
    else:
        intrinsic = max(K - S, 0)
    
    # If option price is less than intrinsic value, return NaN
    if option_price < intrinsic * 0.99:  # Allow small rounding errors
        return np.nan
    
    # Use Brenner-Subrahmanyam approximation for ATM options as initial guess
    # IV ≈ sqrt(2*pi/T) * (option_price / S)
    if abs(S - K) / S < 0.1:  # Near ATM
        iv_guess = np.sqrt(2 * np.pi / T) * (option_price / S)
        iv_guess = max(0.05, min(3.0, iv_guess))  # Clamp between 5% and 300%
    else:
        # For non-ATM, use simple guess based on moneyness
        moneyness = S / K if option_type == 'C' else K / S
        if moneyness > 1.2:  # Deep ITM
            iv_guess = 0.3
        elif moneyness < 0.8:  # Deep OTM
            iv_guess = 0.5
        else:
            iv_guess = 0.35
    
    def objective(sigma):
        if option_type == 'C':
            return black_scholes_call(S, K, T, r, sigma) - option_price
        else:
            return black_scholes_put(S, K, T, r, sigma) - option_price
    
    try:
        # Use narrower bounds based on initial guess
        lower_bound = max(0.01, iv_guess * 0.3)
        upper_bound = min(5.0, iv_guess * 3.0)
        
        # First try with narrow bounds
        iv = brentq(objective, lower_bound, upper_bound, maxiter=50, xtol=0.0001)
        return iv
    except:
        try:
            # Fallback to wider bounds
            iv = brentq(objective, 0.01, 5.0, maxiter=100, xtol=0.001)
            return iv
        except:
            return np.nan

def calculate_parkinson_volatility(high, low, window=20):
    """
    Calculate Parkinson volatility estimator
    Parkinson = sqrt(1/(4*ln(2)) * mean((ln(High/Low))^2))
    More efficient than close-to-close volatility
    """
    log_hl = np.log(high / low)
    parkinson = np.sqrt(1 / (4 * np.log(2)) * (log_hl ** 2).rolling(window=window).mean())
    return parkinson

# ============================================================================
# DATA RETRIEVAL FUNCTIONS
# ============================================================================

def get_risk_free_rate(start_date, end_date, api_key):
    """
    Retrieve risk-free rate (10-year Treasury) from FRED
    """
    print("Retrieving risk-free rate from FRED...")
    try:
        fred = Fred(api_key=api_key)
        # DGS10 = 10-year Treasury Constant Maturity Rate
        rates = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)
        rates_df = pd.DataFrame({'date': rates.index, 'risk_free_rate': rates.values / 100})  # Convert to decimal
        rates_df['date'] = pd.to_datetime(rates_df['date'])
        
        # Forward fill missing values (weekends, holidays)
        rates_df = rates_df.set_index('date').asfreq('D', method='ffill').reset_index()
        
        print(f"Retrieved {len(rates_df)} days of risk-free rate data")
        return rates_df
    except Exception as e:
        print(f"Error retrieving risk-free rate: {e}")
        print("Using default rate of 4.5%")
        # Fallback to constant rate
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({'date': date_range, 'risk_free_rate': 0.045})

def get_stock_data(tickers, start_date, end_date):
    """
    Retrieve stock price data from Yahoo Finance
    """
    print(f"\nRetrieving stock data for {len(tickers)} tickers...")
    all_stock_data = []
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"  [{i+1}/{len(tickers)}] Downloading {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"    Warning: No data for {ticker}")
                continue
            
            df = df.reset_index()
            df['ticker'] = ticker
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Select relevant columns
            df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            all_stock_data.append(df)
            
        except Exception as e:
            print(f"    Error downloading {ticker}: {e}")
            continue
    
    if not all_stock_data:
        raise ValueError("No stock data retrieved!")
    
    stock_df = pd.concat(all_stock_data, ignore_index=True)
    print(f"\nTotal stock records: {len(stock_df)}")
    return stock_df

def get_options_data_from_local(options_path, tickers_filter):
    """
    Load options data from local gzipped CSV files
    """
    print(f"\nLoading options data from local directory: {options_path}")
    
    # Check if directory exists
    if not os.path.exists(options_path):
        raise ValueError(f"Options data directory not found: {options_path}")
    
    # Get all .gz files in the directory (handle both .csv.gz and _csv.gz formats)
    all_files = [f for f in os.listdir(options_path) if f.endswith('.gz')]
    
    if not all_files:
        raise ValueError(f"No .gz files found in {options_path}")
    
    # Filter to only December 2025 files for faster processing
    december_files = [f for f in all_files if '2025-12-' in f]
    
    if not december_files:
        print(f"Warning: No December 2025 files found. Processing all {len(all_files)} files...")
        december_files = all_files
    else:
        print(f"Found {len(december_files)} December 2025 files (out of {len(all_files)} total)")
        all_files = december_files
    
    # Create a mapping for tickers with special characters
    ticker_mapping = {
        'BRK.B': 'BRKB',
    }
    
    # Create reverse mapping and combined filter
    reverse_mapping = {v: k for k, v in ticker_mapping.items()}
    tickers_filter_extended = set(tickers_filter)
    for ticker in tickers_filter:
        if ticker in ticker_mapping:
            tickers_filter_extended.add(ticker_mapping[ticker])
    
    all_options_data = []
    successful_files = 0
    failed_files = 0
    
    # Sort files by name to process chronologically
    all_files.sort()
    
    for i, filename in enumerate(all_files):
        file_path = os.path.join(options_path, filename)
        
        try:
            # Extract date from filename
            # Handle formats like: 2025-01-02_csv.gz, 2025-01-02.csv.gz, etc.
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                date_str = date_match.group(1)
                trade_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                # If we can't parse the date from filename, skip
                if failed_files < 5:
                    print(f"  Warning: Could not parse date from filename: {filename}")
                failed_files += 1
                continue
            
            # Read the gzipped CSV file
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
            
            # Check if the dataframe is empty
            if df.empty:
                continue
            
            # Parse tickers and filter for S&P 100 stocks
            # The ticker column has format like "O:A250117C00125000"
            df['parsed'] = df['ticker'].apply(parse_option_ticker)
            df = df[df['parsed'].notna()]
            
            if len(df) > 0:
                df['underlying'] = df['parsed'].apply(lambda x: x['underlying'])
                
                # Apply ticker mapping for filtering
                df['underlying_mapped'] = df['underlying'].apply(
                    lambda x: reverse_mapping.get(x, x)
                )
                
                # Filter for our tickers
                df = df[df['underlying_mapped'].isin(tickers_filter)]
                
                if not df.empty:
                    # Extract option details
                    df['expiration'] = df['parsed'].apply(lambda x: x['expiration'])
                    df['option_type'] = df['parsed'].apply(lambda x: x['option_type'])
                    df['strike'] = df['parsed'].apply(lambda x: x['strike'])
                    df['trade_date'] = trade_date
                    
                    # Use the mapped underlying ticker for consistency with stock data
                    df['underlying'] = df['underlying_mapped']
                    
                    # Select relevant columns
                    df = df[['underlying', 'trade_date', 'option_type', 'strike', 'expiration',
                            'volume', 'open', 'close', 'high', 'low', 'transactions']]
                    
                    all_options_data.append(df)
                    successful_files += 1
                    
                    if successful_files <= 5:  # Print details for first few successful files
                        print(f"  ✓ {filename}: Found {len(df)} options for {df['underlying'].nunique()} tickers")
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(all_files)} files processed, {successful_files} with S&P 100 data")
        
        except Exception as e:
            failed_files += 1
            if failed_files <= 5:  # Print first few errors for debugging
                print(f"  ✗ {filename}: Error - {str(e)[:100]}")
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(all_files)} files processed, {successful_files} with S&P 100 data, {failed_files} failed")
            continue
    
    print(f"\nSuccessfully loaded options data from {successful_files} files")
    print(f"Failed/empty: {failed_files} files")
    
    if not all_options_data:
        raise ValueError("No options data retrieved! Check if files contain data for S&P 100 tickers.")
    
    options_df = pd.concat(all_options_data, ignore_index=True)
    print(f"Total options records: {len(options_df)}")
    print(f"Unique underlyings: {options_df['underlying'].nunique()}")
    print(f"Tickers found: {sorted(options_df['underlying'].unique())}")
    print(f"Date range: {options_df['trade_date'].min()} to {options_df['trade_date'].max()}")
    
    return options_df

# ============================================================================
# DATA CLEANING AND PROCESSING
# ============================================================================

def clean_stock_data(stock_df):
    """
    Clean stock data: handle missing values, outliers
    """
    print("\nCleaning stock data...")
    initial_count = len(stock_df)
    
    # Remove rows with missing OHLC data
    stock_df = stock_df.dropna(subset=['open', 'high', 'low', 'close'])
    
    # Remove rows where high < low (data errors)
    stock_df = stock_df[stock_df['high'] >= stock_df['low']]
    
    # Remove rows with zero or negative prices
    stock_df = stock_df[(stock_df['open'] > 0) & (stock_df['high'] > 0) & 
                       (stock_df['low'] > 0) & (stock_df['close'] > 0)]
    
    # Sort by ticker and date
    stock_df = stock_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"  Removed {initial_count - len(stock_df)} invalid records")
    print(f"  Final stock records: {len(stock_df)}")
    
    return stock_df

def clean_options_data(options_df):
    """
    Clean options data: handle missing values, filter anomalies
    """
    print("\nCleaning options data...")
    initial_count = len(options_df)
    
    # Remove rows with missing price data
    options_df = options_df.dropna(subset=['open', 'close', 'high', 'low'])
    
    # Remove rows where high < low
    options_df = options_df[options_df['high'] >= options_df['low']]
    
    # Remove zero or negative prices
    options_df = options_df[(options_df['close'] > 0) & (options_df['high'] > 0)]
    
    # Remove options with very low volume (likely stale quotes)
    options_df = options_df[options_df['volume'] >= 5]
    
    # Remove options with strikes <= 0
    options_df = options_df[options_df['strike'] > 0]
    
    # Calculate time to expiration
    options_df['trade_date_dt'] = pd.to_datetime(options_df['trade_date'])
    options_df['expiration_dt'] = pd.to_datetime(options_df['expiration'])
    options_df['days_to_expiration'] = (options_df['expiration_dt'] - options_df['trade_date_dt']).dt.days
    
    # Remove options expiring today or in the past (< 1 day)
    options_df = options_df[options_df['days_to_expiration'] >= 1]
    
    # Remove options expiring too far out (> 365 days)
    options_df = options_df[options_df['days_to_expiration'] <= 365]
    
    # Calculate bid-ask spread proxy (high - low) / close
    options_df['spread_pct'] = (options_df['high'] - options_df['low']) / options_df['close']
    
    # Remove options with extreme spreads (> 50%)
    options_df = options_df[options_df['spread_pct'] <= 0.5]
    
    # Sort by underlying, date, expiration
    options_df = options_df.sort_values(['underlying', 'trade_date', 'expiration', 'strike']).reset_index(drop=True)
    
    print(f"  Removed {initial_count - len(options_df)} invalid records")
    print(f"  Final options records: {len(options_df)}")
    
    return options_df

def calculate_implied_volatilities(options_df, stock_df, risk_free_df):
    """
    Calculate implied volatility for each option
    """
    print("\nCalculating implied volatilities...")
    
    # Merge options with stock prices
    options_df['date'] = options_df['trade_date']
    stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
    
    merged = options_df.merge(
        stock_df[['ticker', 'date', 'close']],
        left_on=['underlying', 'trade_date'],
        right_on=['ticker', 'date'],
        how='left'
    )
    
    merged = merged.rename(columns={'close_x': 'option_close', 'close_y': 'stock_close'})
    
    # Merge with risk-free rate
    risk_free_df['date'] = pd.to_datetime(risk_free_df['date']).dt.date
    merged = merged.merge(risk_free_df, left_on='trade_date', right_on='date', how='left')
    
    # Remove rows where we don't have stock price
    initial_count = len(merged)
    merged = merged.dropna(subset=['stock_close', 'risk_free_rate'])
    print(f"  Removed {initial_count - len(merged)} options without matching stock prices")
    
    if len(merged) == 0:
        raise ValueError("No options with matching stock prices! Check date alignment.")
    
    print(f"  Computing IV for {len(merged)} options (optimized)...")
    
    # Pre-filter obviously bad options to speed up processing
    merged = merged[
        (merged['option_close'] > 0) & 
        (merged['stock_close'] > 0) & 
        (merged['strike'] > 0) &
        (merged['days_to_expiration'] > 0)
    ]
    
    print(f"  After pre-filtering: {len(merged)} options to process")
    
    # Calculate IV with progress tracking
    ivs = []
    batch_size = 50000  # Larger batches for better performance
    
    for i in range(0, len(merged), batch_size):
        batch = merged.iloc[i:i+batch_size]
        
        # Use apply with the optimized function
        batch_ivs = batch.apply(
            lambda row: implied_volatility_fast(
                row['option_close'],
                row['stock_close'],
                row['strike'],
                row['days_to_expiration'] / 365.0,
                row['risk_free_rate'],
                row['option_type']
            ),
            axis=1
        )
        
        ivs.extend(batch_ivs.tolist())
        
        print(f"    Processed {min(i + batch_size, len(merged))}/{len(merged)} options")
    
    merged['implied_volatility'] = ivs
    
    # Remove options with failed IV calculation
    valid_iv_count = merged['implied_volatility'].notna().sum()
    merged = merged.dropna(subset=['implied_volatility'])
    
    # Remove extreme IVs (> 300%)
    extreme_iv_count = (merged['implied_volatility'] > 3.0).sum()
    merged = merged[merged['implied_volatility'] <= 3.0]
    
    print(f"  Successfully calculated IV for {len(merged)} options")
    print(f"  Valid IVs: {valid_iv_count}, Removed extreme IVs (>300%): {extreme_iv_count}")
    
    return merged

def calculate_target_variable(stock_df, forward_window=5, volatility_window=10):
    """
    Calculate target variable: future Parkinson volatility
    Note: Using smaller windows (5 days forward, 10 days volatility) to work with December-only data
    """
    print(f"\nCalculating target variable (Parkinson volatility)...")
    print(f"  Using forward window: {forward_window} days, volatility window: {volatility_window} days")
    
    target_data = []
    
    for ticker in stock_df['ticker'].unique():
        ticker_df = stock_df[stock_df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)
        
        # Calculate historical Parkinson volatility
        ticker_df['historical_vol'] = calculate_parkinson_volatility(
            ticker_df['high'], 
            ticker_df['low'], 
            window=volatility_window
        )
        
        # Calculate future Parkinson volatility (shifted backward)
        ticker_df['future_vol'] = ticker_df['historical_vol'].shift(-forward_window)
        
        # Calculate volatility change
        ticker_df['vol_change'] = ticker_df['future_vol'] - ticker_df['historical_vol']
        ticker_df['vol_change_pct'] = ticker_df['vol_change'] / ticker_df['historical_vol']
        
        target_data.append(ticker_df)
    
    target_df = pd.concat(target_data, ignore_index=True)
    
    # Remove rows without valid target (end of time series)
    initial_count = len(target_df)
    target_df = target_df.dropna(subset=['future_vol', 'historical_vol'])
    print(f"  Removed {initial_count - len(target_df)} records without valid target")
    print(f"  Target variable calculated for {len(target_df)} records")
    
    return target_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("VOLATILITY PREDICTION MODEL - DATA RETRIEVAL AND CLEANUP")
    print("=" * 80)
    
    # Step 1: Get risk-free rate
    risk_free_df = get_risk_free_rate(START_DATE, END_DATE, FRED_API_KEY)
    
    # Step 2: Get stock data
    stock_df = get_stock_data(SP100_TICKERS, START_DATE, END_DATE)
    
    # Get list of tickers that actually have data
    available_tickers = stock_df['ticker'].unique().tolist()
    print(f"\nSuccessfully retrieved data for {len(available_tickers)} tickers")
    
    # Step 3: Get options data from local files
    options_df = get_options_data_from_local(OPTIONS_DATA_PATH, available_tickers)
    
    # Step 4: Clean data
    stock_df = clean_stock_data(stock_df)
    options_df = clean_options_data(options_df)
    
    # Step 5: Calculate implied volatilities
    options_with_iv = calculate_implied_volatilities(options_df, stock_df, risk_free_df)
    
    # Step 6: Calculate target variable
    stock_with_target = calculate_target_variable(stock_df)
    
    # Step 7: Save processed data
    print("\nSaving processed data...")
    
    stock_with_target.to_csv(os.path.join(PROCESSED_DATA_DIR, 'stock_data_with_target.csv'), index=False)
    options_with_iv.to_csv(os.path.join(PROCESSED_DATA_DIR, 'options_data_with_iv.csv'), index=False)
    risk_free_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'risk_free_rate.csv'), index=False)
    
    print(f"  Saved stock data: {len(stock_with_target)} records")
    print(f"  Saved options data: {len(options_with_iv)} records")
    print(f"  Saved risk-free rate: {len(risk_free_df)} records")
    
    # Step 8: Display summary statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    
    print(f"\nStock Data:")
    print(f"  Number of tickers: {stock_with_target['ticker'].nunique()}")
    print(f"  Date range: {stock_with_target['date'].min()} to {stock_with_target['date'].max()}")
    print(f"  Total records: {len(stock_with_target)}")
    print(f"\n  Sample statistics:")
    print(stock_with_target[['open', 'high', 'low', 'close', 'historical_vol', 'future_vol']].describe())
    
    print(f"\nOptions Data:")
    print(f"  Number of underlying tickers: {options_with_iv['underlying'].nunique()}")
    print(f"  Date range: {options_with_iv['trade_date'].min()} to {options_with_iv['trade_date'].max()}")
    print(f"  Total records: {len(options_with_iv)}")
    print(f"  Calls vs Puts: {options_with_iv['option_type'].value_counts().to_dict()}")
    print(f"\n  Implied Volatility statistics:")
    print(options_with_iv['implied_volatility'].describe())
    
    # Step 9: Display sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    
    print("\nStock Data Sample (first 5 rows):")
    print(stock_with_target.head())
    
    print("\nOptions Data Sample (first 5 rows):")
    print(options_with_iv[['underlying', 'trade_date', 'option_type', 'strike', 'expiration', 
                           'stock_close', 'option_close', 'implied_volatility']].head())
    
    print("\n" + "=" * 80)
    print("DATA RETRIEVAL AND CLEANUP COMPLETE!")
    print("=" * 80)
    
    return stock_with_target, options_with_iv, risk_free_df

if __name__ == "__main__":
    stock_data, options_data, risk_free_data = main()