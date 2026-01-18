"""
VOLATILITY PREDICTION 
===============================================

Complete end-to-end volatility prediction system with all 76 features.
Loads model once, predicts many times efficiently.

Usage:
    from getprediction import predict_volatility
    
    prediction = predict_volatility('AAPL', api_key='your_key', model_dir='./models')
    print(f"Volatility will: {prediction['prediction']}")
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from datetime import datetime, timedelta
from massive import RESTClient
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL DEPLOYMENT CLASS
# ============================================================================

class VolatilityPredictor:
    """
    Loads and manages the trained volatility prediction model
    """
    
    def __init__(self, model_dir: str, verbose: bool = False):
        """Load all model artifacts"""
        self.verbose = verbose
        self.model_dir = Path(model_dir)
        
        self._print("Loading model artifacts...")
        
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                f"Please ensure the model directory exists and contains trained model files."
            )
        
        # Required files
        model_path = self.model_dir / 'model_fixed.txt'
        thresholds_path = self.model_dir / 'thresholds_fixed.csv'
        fi_path = self.model_dir / 'feature_importance_fixed.csv'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")
        if not fi_path.exists():
            raise FileNotFoundError(f"Feature importance file not found: {fi_path}")
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load thresholds
        thresholds_df = pd.read_csv(thresholds_path)
        self.thresholds = thresholds_df['threshold'].tolist()
        
        # Load feature names
        fi_df = pd.read_csv(fi_path)
        self.feature_names = fi_df['feature'].tolist()
        
        # Try to load scaler and imputation dict
        scaler_path = self.model_dir / 'scaler_fixed.pkl'
        imputation_path = self.model_dir / 'imputation_dict_fixed.pkl'
        
        if scaler_path.exists() and imputation_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(imputation_path, 'rb') as f:
                self.imputation_dict = pickle.load(f)
            self._print(f"  ✓ Model loaded with preprocessing ({self.model.num_trees()} trees, {len(self.feature_names)} features)")
        else:
            self._print(f"  ⚠ Warning: scaler/imputation files not found")
            self._print(f"    Will compute from data (retrain model to save these)")
            self.scaler = None
            self.imputation_dict = None
            self._print(f"  ✓ Model loaded ({self.model.num_trees()} trees, {len(self.feature_names)} features)")
    
    def _print(self, message: str, end: str = '\n'):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message, end=end)
    
    def _prepare_features(self, df):
        """Prepare features exactly as done during training"""
        X = df[self.feature_names].copy()
        
        if self.imputation_dict is None:
            self.imputation_dict = {}
            for col in self.feature_names:
                median_val = X[col].median()
                self.imputation_dict[col] = median_val if not pd.isna(median_val) else 0
        
        for col in self.feature_names:
            X[col].fillna(self.imputation_dict[col], inplace=True)
        
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in self.feature_names:
            X[col].fillna(self.imputation_dict[col], inplace=True)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def _apply_thresholds(self, probabilities):
        """Apply custom thresholds to get predictions"""
        predictions = []
        for proba in probabilities:
            if proba[0] > self.thresholds[0]:
                predictions.append(0)
            elif proba[1] > self.thresholds[1]:
                predictions.append(1)
            elif proba[2] > self.thresholds[2]:
                predictions.append(2)
            else:
                predictions.append(np.argmax(proba))
        return np.array(predictions)
    
    def predict(self, df):
        """Make predictions"""
        X_scaled = self._prepare_features(df)
        probabilities = self.model.predict(X_scaled)
        predictions = self._apply_thresholds(probabilities)
        return predictions, probabilities
    
    def predict_with_labels(self, df):
        """Get predictions with human-readable labels"""
        predictions, probabilities = self.predict(df)
        
        label_map = {0: 'decrease', 1: 'stay_same', 2: 'increase'}
        
        return pd.DataFrame({
            'prediction': [label_map[p] for p in predictions],
            'prediction_numeric': predictions,
            'prob_decrease': probabilities[:, 0],
            'prob_stay_same': probabilities[:, 1],
            'prob_increase': probabilities[:, 2],
            'confidence': np.max(probabilities, axis=1)
        })


# ============================================================================
# REAL-TIME DATA FETCHER
# ============================================================================

class DataFetcher:
    """Efficiently fetches and caches market data"""
    
    def __init__(self, massive_api_key: str, verbose: bool = False):
        """Initialize data fetcher"""
        self.massive_client = RESTClient(massive_api_key)
        self.verbose = verbose
        self._market_cache = {}
        self._cache_timestamp = None
    
    def _print(self, message: str, end: str = '\n'):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message, end=end)
    
    def fetch_stock_data(self, ticker: str, days: int = 25):
        """Fetch stock data from Yahoo Finance"""
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No stock data found for {ticker}")
        
        df.columns = [col.lower() for col in df.columns]
        return df
    
    def fetch_options_data(self, ticker: str):
        """Fetch options chain from massive.com"""
        options_chain = []
        try:
            for option in self.massive_client.list_snapshot_options_chain(
                ticker,
                params={"order": "asc", "limit": 250}
            ):
                options_chain.append(option)
        except Exception as e:
            self._print(f"    ⚠ Options data unavailable: {e}")
        return options_chain
    
    def fetch_market_data(self):
        """Fetch market data (VIX, SPY) with caching and ALL required features"""
        now = datetime.now()
        if (self._cache_timestamp is None or 
            (now - self._cache_timestamp).seconds > 60):
            
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="15d")  # Extended to 15 days for moving averages
            
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="15d")  # Extended for better calculations
            
            # VIX features
            if not vix_data.empty:
                vix_close = float(vix_data['Close'].iloc[-1])
                vix_high = float(vix_data['High'].iloc[-1])
                vix_low = float(vix_data['Low'].iloc[-1])
                
                # VIX changes
                vix_change_1d = float(vix_data['Close'].diff().iloc[-1])
                vix_change_pct_1d = float(vix_data['Close'].pct_change().iloc[-1])
                
                # VIX moving averages
                vix_ma_5d = float(vix_data['Close'].rolling(5).mean().iloc[-1])
                vix_ma_10d = float(vix_data['Close'].rolling(10).mean().iloc[-1])
                
                # VIX regimes
                vix_regime_high = 1 if vix_close > 20 else 0
                vix_regime_extreme = 1 if vix_close > 30 else 0
            else:
                vix_close = 16.0
                vix_high = 17.0
                vix_low = 15.0
                vix_change_1d = 0.0
                vix_change_pct_1d = 0.0
                vix_ma_5d = 16.0
                vix_ma_10d = 16.0
                vix_regime_high = 0
                vix_regime_extreme = 0
            
            # SPY features
            if not spy_data.empty:
                spy_close = float(spy_data['Close'].iloc[-1])
                spy_volume = float(spy_data['Volume'].iloc[-1])
                
                # SPY returns
                spy_returns = spy_data['Close'].pct_change()
                market_return_1d = float(spy_returns.iloc[-1])
                market_return_5d = float(spy_data['Close'].pct_change(5).iloc[-1])
                
                # SPY volatility
                market_vol_5d = float(spy_returns.rolling(5).std().iloc[-1] * np.sqrt(252))
                market_vol_10d = float(spy_returns.rolling(10).std().iloc[-1] * np.sqrt(252))
            else:
                spy_close = 480.0
                spy_volume = 50000000
                market_return_1d = 0.0
                market_return_5d = 0.0
                market_vol_5d = 0.15
                market_vol_10d = 0.15
            
            self._market_cache = {
                'vix_close': vix_close,
                'vix_high': vix_high,
                'vix_low': vix_low,
                'vix_change_1d': vix_change_1d,
                'vix_change_pct_1d': vix_change_pct_1d,
                'vix_ma_5d': vix_ma_5d,
                'vix_ma_10d': vix_ma_10d,
                'vix_regime_high': vix_regime_high,
                'vix_regime_extreme': vix_regime_extreme,
                'spy_close': spy_close,
                'spy_volume': spy_volume,
                'market_return_1d': market_return_1d,
                'market_return_5d': market_return_5d,
                'market_vol_5d': market_vol_5d,
                'market_vol_10d': market_vol_10d
            }
            self._cache_timestamp = now
        
        return self._market_cache


# ============================================================================
# FEATURE CALCULATOR (COMPLETE WITH ALL 76 FEATURES)
# ============================================================================

class FeatureCalculator:
    """
    Calculate ALL 76 features that the model expects
    """
    
    @staticmethod
    def calculate_stock_features(stock_df):
        """Calculate ALL stock-based features (matching training exactly)"""
        close = stock_df['close']
        high = stock_df['high']
        low = stock_df['low']
        volume = stock_df['volume']
        
        # Pre-calculate common values
        returns = close.pct_change()
        log_returns = np.log(close / close.shift(1))
        log_hl = np.log(high / low)
        log_co = np.log(close / close.shift(1))
        
        features = {}
        
        # ===== RETURNS (multiple periods) =====
        features['return_1d'] = float(returns.iloc[-1]) if not pd.isna(returns.iloc[-1]) else 0.0
        features['return_5d'] = float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else 0.0
        features['return_10d'] = float(close.pct_change(10).iloc[-1]) if len(close) >= 11 else 0.0
        features['return_acceleration'] = float(returns.diff().iloc[-1]) if len(returns) >= 2 else 0.0
        
        # ===== INTRADAY RANGE =====
        features['intraday_range'] = float((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1])
        
        # ===== VOLATILITY (multiple windows and estimators) =====
        for window in [5, 10, 15]:
            if len(log_returns) >= window + 1:
                # Close-to-close volatility
                vol_close = log_returns.rolling(window).std().iloc[-1] * np.sqrt(252)
                features[f'vol_close_{window}d'] = float(vol_close) if not pd.isna(vol_close) else 0.15
                
                # Parkinson volatility
                vol_park = np.sqrt(
                    1 / (4 * np.log(2)) * (log_hl ** 2).rolling(window).mean().iloc[-1]
                ) * np.sqrt(252)
                features[f'vol_parkinson_{window}d'] = float(vol_park) if not pd.isna(vol_park) else 0.15
                
                # Garman-Klass volatility
                vol_gk = np.sqrt(
                    0.5 * (log_hl ** 2).rolling(window).mean().iloc[-1] - 
                    (2 * np.log(2) - 1) * (log_co ** 2).rolling(window).mean().iloc[-1]
                ) * np.sqrt(252)
                features[f'vol_gk_{window}d'] = float(vol_gk) if not pd.isna(vol_gk) else 0.15
                
                # Volatility of volatility
                vol_series = log_returns.rolling(window).std() * np.sqrt(252)
                vol_of_vol = vol_series.rolling(5).std().iloc[-1]
                features[f'vol_of_vol_{window}d'] = float(vol_of_vol) if not pd.isna(vol_of_vol) else 0.01
            else:
                features[f'vol_close_{window}d'] = 0.15
                features[f'vol_parkinson_{window}d'] = 0.15
                features[f'vol_gk_{window}d'] = 0.15
                features[f'vol_of_vol_{window}d'] = 0.01
        
        # ===== RSI =====
        features['rsi'] = FeatureCalculator._calculate_rsi(close, 10)
        
        # ===== MACD (full) =====
        if len(close) >= 15:
            ema_fast = close.ewm(span=8, adjust=False).mean()
            ema_slow = close.ewm(span=15, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=6, adjust=False).mean()
            
            features['macd'] = float(macd_line.iloc[-1])
            features['macd_signal'] = float(macd_signal.iloc[-1])
            features['macd_histogram'] = float((macd_line - macd_signal).iloc[-1])
        else:
            features['macd'] = 0.0
            features['macd_signal'] = 0.0
            features['macd_histogram'] = 0.0
        
        # ===== BOLLINGER BANDS =====
        if len(close) >= 15:
            sma_15 = close.rolling(15).mean()
            std_15 = close.rolling(15).std()
            upper_band = sma_15 + (std_15 * 2)
            lower_band = sma_15 - (std_15 * 2)
            
            bb_pos = ((close.iloc[-1] - lower_band.iloc[-1]) / 
                     (upper_band.iloc[-1] - lower_band.iloc[-1]))
            features['bb_position'] = float(bb_pos) if not pd.isna(bb_pos) else 0.5
            
            bb_w = ((upper_band - lower_band) / sma_15).iloc[-1]
            features['bb_width'] = float(bb_w) if not pd.isna(bb_w) else 0.05
        else:
            features['bb_position'] = 0.5
            features['bb_width'] = 0.05
        
        # ===== MOVING AVERAGE DISTANCE =====
        for window in [5, 10, 15]:
            if len(close) >= window:
                ma = close.rolling(window).mean()
                ma_dist = ((close - ma) / ma).iloc[-1]
                features[f'ma_dist_{window}d'] = float(ma_dist) if not pd.isna(ma_dist) else 0.0
            else:
                features[f'ma_dist_{window}d'] = 0.0
        
        # ===== VOLUME FEATURES =====
        for window in [5, 10]:
            if len(volume) >= window:
                # Volume moving average
                vol_ma = volume.rolling(window).mean()
                features[f'volume_ma_{window}d'] = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else float(volume.iloc[-1])
                
                # Volume ratio
                vol_ratio = (volume / vol_ma).iloc[-1]
                features[f'volume_ratio_{window}d'] = float(vol_ratio) if not pd.isna(vol_ratio) else 1.0
                
                # VWAP ratio
                vwap = (close * volume).rolling(window).sum() / volume.rolling(window).sum()
                vwap_ratio = (close / vwap).iloc[-1]
                features[f'vwap_ratio_{window}d'] = float(vwap_ratio) if not pd.isna(vwap_ratio) else 1.0
            else:
                features[f'volume_ma_{window}d'] = float(volume.iloc[-1])
                features[f'volume_ratio_{window}d'] = 1.0
                features[f'vwap_ratio_{window}d'] = 1.0
        
        # Volume anomaly (z-score)
        if len(volume) >= 10:
            vol_mean = volume.rolling(10).mean().iloc[-1]
            vol_std = volume.rolling(10).std().iloc[-1]
            vol_anom = ((volume.iloc[-1] - vol_mean) / vol_std) if vol_std > 0 else 0
            features['volume_anomaly'] = float(vol_anom) if not pd.isna(vol_anom) else 0.0
        else:
            features['volume_anomaly'] = 0.0
        
        return features
    
    @staticmethod
    def _calculate_rsi(close, period):
        """Calculate RSI"""
        if len(close) < period + 1:
            return 50.0
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        return float(rsi) if not pd.isna(rsi) else 50.0
    
    @staticmethod
    def calculate_options_features(options_chain, stock_price):
        """Calculate ALL options-based features (all 76 total features)"""
        if not options_chain:
            return FeatureCalculator._get_default_options_features()
        
        # Parse options data
        options_list = []
        for opt in options_chain:
            try:
                # Get strike price
                strike = None
                if hasattr(opt, 'details') and opt.details:
                    strike = getattr(opt.details, 'strike_price', None)
                
                if not strike:
                    continue
                
                # Get other details
                expiration = getattr(opt.details, 'expiration_date', None)
                contract_type = getattr(opt.details, 'contract_type', None)
                
                # Get implied volatility
                iv = getattr(opt, 'implied_volatility', None)
                
                # Get volume
                volume = 0
                if hasattr(opt, 'day') and opt.day:
                    volume = getattr(opt.day, 'volume', 0) or 0
                
                # Get open interest
                open_interest = getattr(opt, 'open_interest', 0) or 0
                
                options_list.append({
                    'strike': float(strike),
                    'expiration': expiration,
                    'type': contract_type,
                    'iv': float(iv) if iv else None,
                    'volume': int(volume),
                    'open_interest': int(open_interest)
                })
            except Exception:
                continue
        
        if not options_list:
            return FeatureCalculator._get_default_options_features()
        
        df = pd.DataFrame(options_list)
        df['moneyness'] = df['strike'] / stock_price
        df['expiration'] = pd.to_datetime(df['expiration'])
        df['dte'] = (df['expiration'] - datetime.now()).dt.days
        
        # Filter valid options
        df = df[(df['dte'] > 0) & (df['dte'] <= 365) & (df['iv'].notna())]
        
        if len(df) == 0:
            return FeatureCalculator._get_default_options_features()
        
        features = {}
        
        # ATM options (95-105% moneyness)
        atm = df[(df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)]
        otm_calls = df[(df['type'] == 'call') & (df['moneyness'] > 1.05)]
        otm_puts = df[(df['type'] == 'put') & (df['moneyness'] < 0.95)]
        itm_options = df[abs(df['moneyness'] - 1.0) > 0.1]
        
        overall_iv = df['iv'].mean()
        
        # ===== ATM IV by expiration =====
        if len(atm) > 0:
            features['atm_iv_short'] = float(atm[atm['dte'] <= 15]['iv'].mean())
            features['atm_iv_medium'] = float(atm[(atm['dte'] > 15) & (atm['dte'] <= 45)]['iv'].mean())
            features['atm_iv_long'] = float(atm[atm['dte'] > 45]['iv'].mean())
            features['atm_iv_std'] = float(atm['iv'].std())
            features['atm_iv_mean'] = float(atm['iv'].mean())
        else:
            features['atm_iv_short'] = overall_iv
            features['atm_iv_medium'] = overall_iv
            features['atm_iv_long'] = overall_iv
            features['atm_iv_std'] = float(df['iv'].std())
            features['atm_iv_mean'] = overall_iv
        
        # Fill NaN with defaults
        for key in ['atm_iv_short', 'atm_iv_medium', 'atm_iv_long', 'atm_iv_std', 'atm_iv_mean']:
            if pd.isna(features[key]):
                features[key] = 0.25
        
        # ===== IV TERM STRUCTURE =====
        if len(atm) >= 2:
            atm_sorted = atm.sort_values('dte')
            iv_near = atm_sorted.iloc[0]['iv']
            iv_far = atm_sorted.iloc[-1]['iv']
            dte_near = atm_sorted.iloc[0]['dte']
            dte_far = atm_sorted.iloc[-1]['dte']
            features['iv_term_structure_slope'] = float((iv_far - iv_near) / (dte_far - dte_near) 
                                                   if dte_far != dte_near else 0)
        else:
            features['iv_term_structure_slope'] = 0.0
        
        # ===== IV SKEW =====
        features['iv_skew_call'] = float(otm_calls['iv'].mean() - features['atm_iv_mean'] 
                                    if len(otm_calls) > 0 else 0.0)
        features['iv_skew_put'] = float(otm_puts['iv'].mean() - features['atm_iv_mean']
                                   if len(otm_puts) > 0 else 0.0)
        
        # ===== IV SMILE =====
        features['iv_smile'] = float(otm_puts['iv'].mean() - otm_calls['iv'].mean() 
                               if len(otm_puts) > 0 and len(otm_calls) > 0 else 0.0)
        
        # ===== IV STATISTICS =====
        features['iv_p25'] = float(df['iv'].quantile(0.25))
        features['iv_p50'] = float(df['iv'].quantile(0.50))
        features['iv_p75'] = float(df['iv'].quantile(0.75))
        features['iv_range'] = float(df['iv'].max() - df['iv'].min())
        
        # ===== VOLUME FEATURES =====
        features['total_option_volume'] = float(df['volume'].sum())
        features['avg_option_volume'] = float(df['volume'].mean())
        features['total_transactions'] = float(len(df))
        
        # Put/Call ratios
        put_volume = df[df['type'] == 'put']['volume'].sum()
        call_volume = df[df['type'] == 'call']['volume'].sum()
        features['put_call_volume_ratio'] = float(put_volume / call_volume if call_volume > 0 else 1.0)
        
        # ===== VOLUME DISTRIBUTION =====
        atm_vol = atm['volume'].sum() if len(atm) > 0 else 0
        itm_vol = itm_options['volume'].sum()
        total_vol = df['volume'].sum()
        
        features['volume_pct_atm'] = float(atm_vol / total_vol if total_vol > 0 else 0.5)
        features['volume_pct_itm'] = float(itm_vol / total_vol if total_vol > 0 else 0.3)
        
        # ===== TIME-SERIES FEATURES (defaults to 0 - require historical data) =====
        features['atm_iv_mean_change_1d'] = 0.0
        features['atm_iv_mean_change_5d'] = 0.0
        features['put_call_volume_ratio_change_1d'] = 0.0
        features['put_call_volume_ratio_change_5d'] = 0.0
        features['iv_skew_put_change_1d'] = 0.0
        features['iv_skew_put_change_5d'] = 0.0
        features['vol_risk_premium'] = float(features['atm_iv_mean'] - 0.25)
        features['vol_risk_premium_change_1d'] = 0.0
        
        return features
    
    @staticmethod
    def _get_default_options_features():
        """Default values when options data unavailable"""
        return {
            'atm_iv_short': 0.25, 'atm_iv_medium': 0.26, 'atm_iv_long': 0.28,
            'atm_iv_std': 0.04, 'atm_iv_mean': 0.25, 'iv_skew_call': 0.02,
            'iv_skew_put': 0.02, 'iv_smile': 0.02, 'iv_p25': 0.22,
            'iv_p50': 0.25, 'iv_p75': 0.28, 'iv_range': 0.15,
            'total_option_volume': 50000.0, 'avg_option_volume': 5000.0,
            'put_call_volume_ratio': 0.85, 'volume_pct_atm': 0.35,
            'volume_pct_itm': 0.30, 'total_transactions': 5000.0,
            'atm_iv_mean_change_1d': 0.0, 'atm_iv_mean_change_5d': 0.0,
            'put_call_volume_ratio_change_1d': 0.0, 'put_call_volume_ratio_change_5d': 0.0,
            'iv_skew_put_change_1d': 0.0, 'iv_skew_put_change_5d': 0.0,
            'vol_risk_premium': 0.0, 'vol_risk_premium_change_1d': 0.0,
            'iv_term_structure_slope': 0.0
        }


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class VolatilityPredictionEngine:
    """Main prediction engine that coordinates all components"""
    
    def __init__(self, model_dir: str, massive_api_key: str, verbose: bool = False):
        """Initialize the prediction engine"""
        self.verbose = verbose
        self._print("Initializing Volatility Prediction Engine...")
        
        self.predictor = VolatilityPredictor(model_dir, verbose=verbose)
        self.data_fetcher = DataFetcher(massive_api_key, verbose=verbose)
        self.feature_calculator = FeatureCalculator()
        
        self._print("✓ Ready!\n")
    
    def _print(self, message: str, end: str = '\n'):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message, end=end)
    
    def predict(self, ticker: str) -> Dict:
        """Make a volatility prediction for a stock"""
        if self.verbose:
            self._print("=" * 80)
            self._print(f"VOLATILITY PREDICTION FOR {ticker}")
            self._print("=" * 80)
        
        # Fetch data
        self._print("\nFetching data...")
        
        try:
            stock_df = self.data_fetcher.fetch_stock_data(ticker)
            options_chain = self.data_fetcher.fetch_options_data(ticker)
            market_data = self.data_fetcher.fetch_market_data()
        except Exception as e:
            error_msg = f"Failed to fetch data: {str(e)}"
            self._print(f"✗ {error_msg}")
            return {
                'error': error_msg,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
        
        if self.verbose:
            self._print(f"  ✓ Stock: {len(stock_df)} days")
            self._print(f"  ✓ Options: {len(options_chain)} contracts")
            self._print(f"  ✓ Market: VIX={market_data['vix_close']:.2f}")
        
        # Calculate features
        self._print("\nCalculating features...")
        
        stock_features = self.feature_calculator.calculate_stock_features(stock_df)
        current_price = stock_df['close'].iloc[-1]
        options_features = self.feature_calculator.calculate_options_features(
            options_chain, current_price
        )
        
        # Combine all features
        # Combine all features
        all_features = {**stock_features, **options_features, **market_data}

        # Calculate interaction feature: relative_vol
        if 'vol_close_10d' in all_features and 'market_vol_10d' in market_data:
            all_features['relative_vol'] = (all_features['vol_close_10d'] / 
                                            market_data['market_vol_10d'] if market_data['market_vol_10d'] > 0 else 1.0)
        else:
            all_features['relative_vol'] = 1.0

        features_df = pd.DataFrame([all_features])
        
        self._print(f"  ✓ {len(all_features)} features calculated")
        
        # Make prediction
        self._print("\nMaking prediction...")
        
        try:
            results = self.predictor.predict_with_labels(features_df)
            prediction = results.iloc[0]
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self._print(f"✗ {error_msg}")
            return {
                'error': error_msg,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
        
        # Format output
        output = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'prediction': prediction['prediction'],
            'confidence': float(prediction['confidence']),
            'probabilities': {
                'decrease': float(prediction['prob_decrease']),
                'stay_same': float(prediction['prob_stay_same']),
                'increase': float(prediction['prob_increase'])
            },
            'market_context': {
                'vix': float(market_data['vix_close']),
                'spy': float(market_data['spy_close']),
                'current_iv': float(options_features.get('iv_p50', 0.25))
            }
        }
        
        if self.verbose:
            self._print("\n" + "=" * 80)
            self._print("RESULTS")
            self._print("=" * 80)
            self._print(f"\nTicker: {output['ticker']}")
            self._print(f"Price: ${output['current_price']:.2f}")
            self._print(f"\n>>> PREDICTION: {output['prediction'].upper()} <<<")
            self._print(f"Confidence: {output['confidence']:.1%}")
            self._print(f"\nProbabilities:")
            self._print(f"  Decrease:   {output['probabilities']['decrease']:.1%}")
            self._print(f"  Stay Same:  {output['probabilities']['stay_same']:.1%}")
            self._print(f"  Increase:   {output['probabilities']['increase']:.1%}")
            self._print("=" * 80 + "\n")
        
        return output


# ============================================================================
# SIMPLE API FUNCTIONS
# ============================================================================

def find_model_directory() -> Optional[str]:
    """Try to find the model directory automatically"""
    env_model_dir = os.getenv('MODEL_DIR')
    if env_model_dir and Path(env_model_dir).exists():
        model_file = Path(env_model_dir) / 'model_fixed.txt'
        if model_file.exists():
            return env_model_dir
    
    possible_paths = [
        "./models",
        "./data/processed",
        "../data/processed",
        "../models",
        "./backend/app/volatility/models",
        "../backend/app/volatility/models",
        os.path.expanduser("~/models"),
        os.path.expanduser("~/data/processed"),
    ]
    
    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if (path_obj / 'model_fixed.txt').exists():
                return str(path_obj)
    
    return None


def predict_volatility(
    ticker: str,
    api_key: str,
    model_dir: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """Simple API function to predict volatility change for a stock"""
    if model_dir is None:
        model_dir = find_model_directory()
        if model_dir is None:
            return {
                'error': (
                    "Could not find trained model files. "
                    "Please set MODEL_DIR environment variable or pass model_dir parameter."
                ),
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
    
    try:
        engine = VolatilityPredictionEngine(model_dir, api_key, verbose=verbose)
        return engine.predict(ticker)
    except Exception as e:
        return {
            'error': f"Prediction engine error: {str(e)}",
            'ticker': ticker,
            'timestamp': datetime.now().isoformat()
        }


def predict_volatility_simple(
    ticker: str,
    api_key: str,
    model_dir: Optional[str] = None,
    verbose: bool = False
) -> str:
    """Get just the prediction label"""
    result = predict_volatility(ticker, api_key, model_dir, verbose)
    return result.get('prediction', 'error')


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    API_KEY = os.getenv('POLYGON_API_KEY')
    MODEL_DIR = os.getenv('MODEL_DIR')
    
    if not API_KEY:
        print("Error: POLYGON_API_KEY not found in environment variables.")
        sys.exit(1)
    
    print("VOLATILITY PREDICTION SYSTEM\n")
    
    if not MODEL_DIR:
        print("MODEL_DIR not set, attempting auto-detection...")
        MODEL_DIR = find_model_directory()
        if MODEL_DIR:
            print(f"✓ Found model directory: {MODEL_DIR}\n")
        else:
            print("✗ Could not find model directory.")
            sys.exit(1)
    
    print("Example 1: Simple prediction")
    print("-" * 40)
    result = predict_volatility('AAPL', API_KEY, MODEL_DIR, verbose=False)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nResult: Volatility will {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
