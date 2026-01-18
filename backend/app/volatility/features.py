import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to processed data from Part 1
DATA_DIR = ""
STOCK_DATA_PATH = f"{DATA_DIR}\\stock_data_with_target.csv"
OPTIONS_DATA_PATH = f"{DATA_DIR}\\options_data_with_iv.csv"
RISK_FREE_RATE_PATH = f"{DATA_DIR}\\risk_free_rate.csv"

# Output path for features
FEATURES_OUTPUT_PATH = f"{DATA_DIR}\\features_dataset.csv"

# Feature engineering parameters
SHORT_WINDOW = 5
MEDIUM_WINDOW = 10
LONG_WINDOW = 20

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_returns(prices, periods):
    """Calculate returns over multiple periods"""
    returns = {}
    for period in periods:
        returns[f'return_{period}d'] = prices.pct_change(period)
    return pd.DataFrame(returns)

def calculate_rolling_volatility(high, low, close, windows):
    """Calculate rolling volatility using multiple estimators"""
    vols = {}
    
    for window in windows:
        # Simple close-to-close volatility
        returns = np.log(close / close.shift(1))
        vols[f'vol_close_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (more efficient, uses high-low)
        log_hl = np.log(high / low)
        vols[f'vol_parkinson_{window}d'] = np.sqrt(
            1 / (4 * np.log(2)) * (log_hl ** 2).rolling(window=window).mean()
        ) * np.sqrt(252)
        
        # Garman-Klass volatility (uses OHLC)
        log_hl = np.log(high / low)
        log_co = np.log(close / close.shift(1))
        vols[f'vol_gk_{window}d'] = np.sqrt(
            0.5 * (log_hl ** 2).rolling(window=window).mean() - 
            (2 * np.log(2) - 1) * (log_co ** 2).rolling(window=window).mean()
        ) * np.sqrt(252)
    
    return pd.DataFrame(vols)

def calculate_volatility_of_volatility(volatility, window=10):
    """Calculate how stable the volatility is"""
    return volatility.rolling(window).std()

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    # Note: With only ~20 days of data, using shorter period (10 instead of 14)
    period = min(period, 10)
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD - adjusted for short time series"""
    # Adjust periods for short time series
    fast = min(fast, 8)
    slow = min(slow, 15)
    signal = min(signal, 6)
    
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(close, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    window = min(window, 15)  # Adjust for short time series
    
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    bb_width = (upper_band - lower_band) / sma
    bb_position = (close - lower_band) / (upper_band - lower_band)
    
    return upper_band, lower_band, bb_width, bb_position

def calculate_ma_distance(close, windows):
    """Calculate distance from moving averages"""
    ma_dist = {}
    for window in windows:
        ma = close.rolling(window).mean()
        ma_dist[f'ma_dist_{window}d'] = (close - ma) / ma
    return pd.DataFrame(ma_dist)

def calculate_volume_features(volume, close, windows):
    """Calculate volume-based features"""
    vol_features = {}
    
    for window in windows:
        # Volume moving average
        vol_features[f'volume_ma_{window}d'] = volume.rolling(window).mean()
        
        # Volume ratio (current vs average)
        vol_features[f'volume_ratio_{window}d'] = volume / volume.rolling(window).mean()
        
        # Volume-weighted average price approximation
        vol_features[f'vwap_ratio_{window}d'] = close / (close * volume).rolling(window).sum() / volume.rolling(window).sum()
    
    return pd.DataFrame(vol_features)

def calculate_intraday_range(high, low, close):
    """Calculate intraday range metrics"""
    return (high - low) / close

# ============================================================================
# STOCK-BASED FEATURES
# ============================================================================

def create_stock_features(stock_df):
    """Create all stock-based features"""
    print("\nCreating stock-based features...")
    
    all_features = []
    
    for ticker in stock_df['ticker'].unique():
        print(f"  Processing {ticker}...")
        ticker_df = stock_df[stock_df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)
        
        # Basic info
        features_df = ticker_df[['ticker', 'date', 'close', 'historical_vol', 'future_vol', 
                                 'vol_change', 'vol_change_pct']].copy()
        
        # 1. Returns over multiple periods
        returns = calculate_returns(ticker_df['close'], [1, 5, 10])
        for col in returns.columns:
            features_df[col] = returns[col]
        
        # 2. Rolling volatilities (multiple estimators and windows)
        rolling_vols = calculate_rolling_volatility(
            ticker_df['high'], 
            ticker_df['low'], 
            ticker_df['close'], 
            windows=[5, 10, 15]
        )
        for col in rolling_vols.columns:
            features_df[col] = rolling_vols[col]
        
        # 3. Volatility of volatility
        for window in [5, 10, 15]:
            if f'vol_close_{window}d' in features_df.columns:
                features_df[f'vol_of_vol_{window}d'] = calculate_volatility_of_volatility(
                    features_df[f'vol_close_{window}d'], 
                    window=5
                )
        
        # 4. Price momentum indicators
        features_df['rsi'] = calculate_rsi(ticker_df['close'], period=10)
        
        macd, macd_signal, macd_hist = calculate_macd(ticker_df['close'])
        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_histogram'] = macd_hist
        
        # 5. Bollinger Bands
        bb_upper, bb_lower, bb_width, bb_position = calculate_bollinger_bands(ticker_df['close'], window=15)
        features_df['bb_width'] = bb_width
        features_df['bb_position'] = bb_position
        
        # 6. Distance from moving averages
        ma_distances = calculate_ma_distance(ticker_df['close'], windows=[5, 10, 15])
        for col in ma_distances.columns:
            features_df[col] = ma_distances[col]
        
        # 7. Volume features
        volume_features = calculate_volume_features(
            ticker_df['volume'], 
            ticker_df['close'], 
            windows=[5, 10]
        )
        for col in volume_features.columns:
            features_df[col] = volume_features[col]
        
        # 8. Intraday range
        features_df['intraday_range'] = calculate_intraday_range(
            ticker_df['high'], 
            ticker_df['low'], 
            ticker_df['close']
        )
        
        # 9. Price acceleration (rate of change of returns)
        features_df['return_acceleration'] = features_df['return_1d'].diff()
        
        all_features.append(features_df)
    
    stock_features = pd.concat(all_features, ignore_index=True)
    print(f"  Created {len(stock_features.columns)} stock-based features")
    
    return stock_features

# ============================================================================
# OPTIONS-BASED FEATURES
# ============================================================================

def calculate_moneyness(strike, stock_price):
    """Calculate option moneyness"""
    return strike / stock_price

def find_atm_options(options_df, tolerance=0.05):
    """Find at-the-money options (within tolerance of stock price)"""
    options_df = options_df.copy()
    options_df['moneyness'] = calculate_moneyness(options_df['strike'], options_df['stock_close'])
    
    # ATM if moneyness is close to 1.0
    options_df['is_atm'] = abs(options_df['moneyness'] - 1.0) < tolerance
    
    return options_df

def create_options_features(options_df, stock_dates):
    """Create all options-based features"""
    print("\nCreating options-based features...")
    
    options_df = find_atm_options(options_df)
    
    all_features = []
    
    # Process by ticker and date
    for ticker in options_df['underlying'].unique():
        print(f"  Processing options for {ticker}...")
        ticker_options = options_df[options_df['underlying'] == ticker].copy()
        
        for date in ticker_options['trade_date'].unique():
            date_options = ticker_options[ticker_options['trade_date'] == date].copy()
            
            features = {
                'ticker': ticker,
                'date': date
            }
            
            # 1. ATM Implied Volatility by expiration
            atm_options = date_options[date_options['is_atm']].copy()
            
            if len(atm_options) > 0:
                # Group by days to expiration
                for dte_min, dte_max, label in [(1, 15, 'short'), (15, 45, 'medium'), (45, 120, 'long')]:
                    mask = (atm_options['days_to_expiration'] >= dte_min) & (atm_options['days_to_expiration'] < dte_max)
                    if mask.sum() > 0:
                        features[f'atm_iv_{label}'] = atm_options[mask]['implied_volatility'].mean()
                
                # Overall ATM IV
                features['atm_iv_mean'] = atm_options['implied_volatility'].mean()
                features['atm_iv_std'] = atm_options['implied_volatility'].std()
            
            # 2. IV Term Structure (slope across expirations)
            if len(atm_options) > 1:
                # Sort by days to expiration
                atm_sorted = atm_options.sort_values('days_to_expiration')
                if len(atm_sorted) >= 2:
                    # Calculate slope between nearest and farthest expiration
                    iv_near = atm_sorted.iloc[0]['implied_volatility']
                    iv_far = atm_sorted.iloc[-1]['implied_volatility']
                    dte_near = atm_sorted.iloc[0]['days_to_expiration']
                    dte_far = atm_sorted.iloc[-1]['days_to_expiration']
                    
                    features['iv_term_structure_slope'] = (iv_far - iv_near) / (dte_far - dte_near) if dte_far != dte_near else 0
            
            # 3. IV Skew (OTM puts vs ATM)
            calls = date_options[date_options['option_type'] == 'C'].copy()
            puts = date_options[date_options['option_type'] == 'P'].copy()
            
            # OTM puts (moneyness < 0.95)
            otm_puts = puts[puts['moneyness'] < 0.95]
            if len(otm_puts) > 0 and len(atm_options) > 0:
                features['iv_skew_put'] = otm_puts['implied_volatility'].mean() - atm_options['implied_volatility'].mean()
            
            # OTM calls (moneyness > 1.05)
            otm_calls = calls[calls['moneyness'] > 1.05]
            if len(otm_calls) > 0 and len(atm_options) > 0:
                features['iv_skew_call'] = otm_calls['implied_volatility'].mean() - atm_options['implied_volatility'].mean()
            
            # 4. IV Smile characteristics (OTM puts vs OTM calls)
            if len(otm_puts) > 0 and len(otm_calls) > 0:
                features['iv_smile'] = otm_puts['implied_volatility'].mean() - otm_calls['implied_volatility'].mean()
            
            # 5. Put/Call Volume Ratio
            put_volume = puts['volume'].sum()
            call_volume = calls['volume'].sum()
            features['put_call_volume_ratio'] = put_volume / call_volume if call_volume > 0 else np.nan
            
            # 6. Volume by moneyness buckets
            # ITM (deep in the money)
            itm_options = date_options[abs(date_options['moneyness'] - 1.0) > 0.1]
            atm_vol = date_options[date_options['is_atm']]['volume'].sum()
            itm_vol = itm_options['volume'].sum()
            total_vol = date_options['volume'].sum()
            
            features['volume_pct_atm'] = atm_vol / total_vol if total_vol > 0 else np.nan
            features['volume_pct_itm'] = itm_vol / total_vol if total_vol > 0 else np.nan
            
            # 7. Overall option metrics
            features['total_option_volume'] = date_options['volume'].sum()
            features['avg_option_volume'] = date_options['volume'].mean()
            features['total_transactions'] = date_options['transactions'].sum()
            
            # 8. IV percentiles
            features['iv_p25'] = date_options['implied_volatility'].quantile(0.25)
            features['iv_p50'] = date_options['implied_volatility'].quantile(0.50)
            features['iv_p75'] = date_options['implied_volatility'].quantile(0.75)
            features['iv_range'] = date_options['implied_volatility'].max() - date_options['implied_volatility'].min()
            
            all_features.append(features)
    
    options_features = pd.DataFrame(all_features)
    
    # Calculate time-based features (IV momentum)
    print("  Calculating IV momentum features...")
    options_features = options_features.sort_values(['ticker', 'date'])
    
    for ticker in options_features['ticker'].unique():
        mask = options_features['ticker'] == ticker
        ticker_data = options_features[mask].copy()
        
        # IV changes over time
        for col in ['atm_iv_mean', 'iv_skew_put', 'put_call_volume_ratio']:
            if col in ticker_data.columns:
                options_features.loc[mask, f'{col}_change_1d'] = ticker_data[col].diff(1)
                options_features.loc[mask, f'{col}_change_5d'] = ticker_data[col].diff(5)
    
    print(f"  Created {len(options_features.columns)} options-based features")
    
    return options_features

# ============================================================================
# MARKET REGIME FEATURES
# ============================================================================

def create_market_features(start_date, end_date):
    """Create market-wide regime features"""
    print("\nCreating market regime features...")
    
    # Download VIX data
    print("  Downloading VIX data...")
    vix = yf.Ticker("^VIX")
    vix_df = vix.history(start=start_date, end=end_date)
    
    if not vix_df.empty:
        vix_features = pd.DataFrame({
            'date': vix_df.index.date,
            'vix_close': vix_df['Close'].values,
            'vix_high': vix_df['High'].values,
            'vix_low': vix_df['Low'].values
        })
        
        # VIX changes
        vix_features['vix_change_1d'] = vix_features['vix_close'].diff(1)
        vix_features['vix_change_pct_1d'] = vix_features['vix_close'].pct_change(1)
        
        # VIX moving averages
        vix_features['vix_ma_5d'] = vix_features['vix_close'].rolling(5).mean()
        vix_features['vix_ma_10d'] = vix_features['vix_close'].rolling(10).mean()
        
        # VIX regime (high vs low)
        vix_features['vix_regime_high'] = (vix_features['vix_close'] > 20).astype(int)
        vix_features['vix_regime_extreme'] = (vix_features['vix_close'] > 30).astype(int)
    else:
        # Create empty dataframe if VIX data not available
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        vix_features = pd.DataFrame({'date': date_range.date})
    
    # Download SPY (S&P 500) data for market returns
    print("  Downloading SPY data...")
    spy = yf.Ticker("SPY")
    spy_df = spy.history(start=start_date, end=end_date)
    
    if not spy_df.empty:
        spy_features = pd.DataFrame({
            'date': spy_df.index.date,
            'spy_close': spy_df['Close'].values,
            'spy_volume': spy_df['Volume'].values
        })
        
        # Market returns
        spy_features['market_return_1d'] = spy_features['spy_close'].pct_change(1)
        spy_features['market_return_5d'] = spy_features['spy_close'].pct_change(5)
        
        # Market volatility
        spy_features['market_vol_5d'] = spy_features['market_return_1d'].rolling(5).std() * np.sqrt(252)
        spy_features['market_vol_10d'] = spy_features['market_return_1d'].rolling(10).std() * np.sqrt(252)
        
        # Merge with VIX
        market_features = vix_features.merge(spy_features, on='date', how='outer')
    else:
        market_features = vix_features
    
    print(f"  Created {len(market_features.columns)} market regime features")
    
    return market_features

def create_target_labels(df, threshold=0.15):
    """
    Create classification labels for volatility prediction:
    - 'increase': future_vol > current_vol * (1 + threshold)
    - 'decrease': future_vol < current_vol * (1 - threshold)  
    - 'stay_same': otherwise
    
    Args:
        df: DataFrame with 'historical_vol' and 'future_vol' columns
        threshold: Threshold for classifying as increase/decrease (default 15%)
    
    Returns:
        DataFrame with added 'target_label' column
    """
    print(f"\nCreating target labels with threshold={threshold}...")
    
    df = df.copy()
    
    # Calculate the change ratio
    df['vol_change_ratio'] = (df['future_vol'] - df['historical_vol']) / df['historical_vol']
    
    # Create labels
    conditions = [
        df['vol_change_ratio'] > threshold,
        df['vol_change_ratio'] < -threshold
    ]
    choices = ['increase', 'decrease']
    
    df['target_label'] = np.select(conditions, choices, default='stay_same')
    
    # Print distribution
    print(f"\nTarget Label Distribution:")
    label_counts = df['target_label'].value_counts()
    print(label_counts)
    print(f"\nPercentages:")
    print(label_counts / len(df) * 100)
    
    # Additional numeric target for potential regression (0=decrease, 1=stay, 2=increase)
    label_map = {'decrease': 0, 'stay_same': 1, 'increase': 2}
    df['target_numeric'] = df['target_label'].map(label_map)
    
    return df

# ============================================================================
# FEATURE COMBINATION AND ENGINEERING
# ============================================================================

def create_combined_features(stock_features, options_features, market_features):
    """Combine all features and create interaction features"""
    print("\nCombining all features...")
    
    # Merge stock and options features
    combined = stock_features.merge(
        options_features,
        on=['ticker', 'date'],
        how='left'
    )
    
    # Merge with market features
    combined['date'] = pd.to_datetime(combined['date']).dt.date
    market_features['date'] = pd.to_datetime(market_features['date']).dt.date
    
    combined = combined.merge(
        market_features,
        on='date',
        how='left'
    )
    
    # Create interaction features
    print("  Creating interaction features...")
    
    # Volatility risk premium (IV - Historical Vol)
    if 'atm_iv_mean' in combined.columns:
        combined['vol_risk_premium'] = combined['atm_iv_mean'] - combined['historical_vol']
        combined['vol_risk_premium_change_1d'] = combined.groupby('ticker')['vol_risk_premium'].diff(1)
    
    # Relative volatility (stock vol vs market vol)
    if 'market_vol_10d' in combined.columns and 'vol_close_10d' in combined.columns:
        combined['relative_vol'] = combined['vol_close_10d'] / combined['market_vol_10d']
    
    # Volume anomaly (current volume vs historical average)
    if 'volume_ma_10d' in combined.columns:
        combined['volume_anomaly'] = combined.groupby('ticker')['volume_ratio_10d'].apply(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    print(f"  Final dataset shape: {combined.shape}")
    print(f"  Total features: {len(combined.columns)}")
    
    return combined

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("VOLATILITY PREDICTION - FEATURE ENGINEERING (PART 2)")
    print("=" * 80)
    
    # Load data from Part 1
    print("\nLoading data from Part 1...")
    stock_df = pd.read_csv(STOCK_DATA_PATH)
    options_df = pd.read_csv(OPTIONS_DATA_PATH)
    
    print(f"  Loaded {len(stock_df)} stock records")
    print(f"  Loaded {len(options_df)} options records")
    
    # Convert date columns
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    options_df['trade_date'] = pd.to_datetime(options_df['trade_date'])
    
    # Get date range
    start_date = stock_df['date'].min()
    end_date = stock_df['date'].max()
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    
    # Create features
    stock_features = create_stock_features(stock_df)
    options_features = create_options_features(options_df, stock_df['date'].unique())
    market_features = create_market_features(start_date, end_date)
    
    # Combine all features
    final_features = create_combined_features(stock_features, options_features, market_features)
    
    # Remove rows without future volatility
    final_features = final_features.dropna(subset=['future_vol'])
    
    # Create target labels (increase/decrease/stay_same)
    final_features = create_target_labels(final_features, threshold=0.15)
    
    # Save features
    print(f"\nSaving features to {FEATURES_OUTPUT_PATH}...")
    final_features.to_csv(FEATURES_OUTPUT_PATH, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 80)
    
    print(f"\nFinal Dataset:")
    print(f"  Shape: {final_features.shape}")
    print(f"  Records: {len(final_features)}")
    print(f"  Features: {len(final_features.columns)}")
    print(f"  Tickers: {final_features['ticker'].nunique()}")
    
    print(f"\nFeature Categories:")
    stock_cols = [col for col in final_features.columns if any(x in col for x in ['return', 'vol_', 'rsi', 'macd', 'bb_', 'ma_dist', 'volume', 'intraday'])]
    options_cols = [col for col in final_features.columns if any(x in col for x in ['iv_', 'atm_', 'put_call', 'skew', 'smile'])]
    market_cols = [col for col in final_features.columns if any(x in col for x in ['vix', 'market', 'spy'])]
    
    print(f"  Stock-based features: {len(stock_cols)}")
    print(f"  Options-based features: {len(options_cols)}")
    print(f"  Market regime features: {len(market_cols)}")
    
    print(f"\nMissing Values by Feature Category:")
    print(f"  Stock features: {final_features[stock_cols].isnull().sum().sum()} / {len(final_features) * len(stock_cols)}")
    print(f"  Options features: {final_features[options_cols].isnull().sum().sum()} / {len(final_features) * len(options_cols)}")
    print(f"  Market features: {final_features[market_cols].isnull().sum().sum()} / {len(final_features) * len(market_cols)}")
    
    print(f"\nSample Features:")
    print(final_features.head())
    
    print(f"\nTarget Variable Distribution:")
    print(final_features['future_vol'].describe())
    
    print(f"\nTarget Labels:")
    print(f"  Increase: {(final_features['target_label'] == 'increase').sum()} ({(final_features['target_label'] == 'increase').sum() / len(final_features) * 100:.1f}%)")
    print(f"  Decrease: {(final_features['target_label'] == 'decrease').sum()} ({(final_features['target_label'] == 'decrease').sum() / len(final_features) * 100:.1f}%)")
    print(f"  Stay Same: {(final_features['target_label'] == 'stay_same').sum()} ({(final_features['target_label'] == 'stay_same').sum() / len(final_features) * 100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    
    return final_features

if __name__ == "__main__":
    features = main()
