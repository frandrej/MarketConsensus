"""
Macro Sensitivity Analysis Tool - API Version
==============================================

Analyzes how sensitive a stock is to macroeconomic factors using multiple 
regression analysis and retrieves relevant news about key macro drivers.

"""

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    print("Note: NewsAPI not available. Install with: pip install newsapi-python")


class MacroSensitivityAnalyzer:
    """
    Analyze stock sensitivity to macroeconomic factors.
    
    Uses multiple regression to quantify relationships between stock returns
    and changes in macro variables. Retrieves relevant news for key drivers.
    """
    
    def __init__(
        self,
        fred_api_key: str,
        news_api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        fred_api_key : str
            API key for FRED (Federal Reserve Economic Data)
            Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
        news_api_key : str, optional
            API key for NewsAPI.org (free tier: 100 requests/day)
            Get at: https://newsapi.org/register
        verbose : bool
            Whether to print progress messages (default: True)
        """
        self.fred = Fred(api_key=fred_api_key)
        self.news_api = NewsApiClient(api_key=news_api_key) if news_api_key and NEWSAPI_AVAILABLE else None
        self.verbose = verbose
        
        # Define macro variables to track
        self.macro_variables = {
            # Interest Rates
            'Federal Funds Rate': 'DFF',
            '10Y Treasury Yield': 'DGS10',
            '2Y Treasury Yield': 'DGS2',
            
            # Inflation
            'CPI (YoY)': 'CPIAUCSL',
            'PCE Inflation': 'PCEPI',
            'PPI': 'PPIACO',
            
            # Employment
            'Unemployment Rate': 'UNRATE',
            'Initial Jobless Claims': 'ICSA',
            
            # Growth
            'GDP Growth': 'GDP',
            'Retail Sales': 'RSXFS',
            'Manufacturing PMI': 'MANEMP',
            
            # Monetary Policy
            'M2 Money Supply': 'M2SL',
            
            # Other (from Yahoo Finance)
            'Oil (WTI)': 'CL=F',
            'Gold': 'GC=F',
            'Dollar Index (DXY)': 'DX-Y.NYB',
            'VIX': '^VIX'
        }
        
        # News keywords for each category
        self.news_keywords = {
            'Federal Funds Rate': 'Federal Reserve interest rate',
            '10Y Treasury Yield': 'treasury yields bonds',
            '2Y Treasury Yield': 'treasury yields short-term',
            'CPI (YoY)': 'CPI inflation consumer prices',
            'PCE Inflation': 'PCE inflation',
            'PPI': 'PPI producer prices inflation',
            'Unemployment Rate': 'unemployment jobs employment',
            'Initial Jobless Claims': 'jobless claims unemployment',
            'GDP Growth': 'GDP economic growth',
            'Retail Sales': 'retail sales consumer spending',
            'Manufacturing PMI': 'manufacturing PMI industrial production',
            'M2 Money Supply': 'money supply monetary policy Fed',
            'Oil (WTI)': 'oil prices crude WTI',
            'Gold': 'gold prices precious metals',
            'Dollar Index (DXY)': 'dollar index currency DXY',
            'VIX': 'VIX volatility market fear'
        }
    
    def _print(self, message: str, end: str = '\n'):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message, end=end)
    
    def fetch_macro_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch all macroeconomic time series data.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with macro variables as columns, dates as index
        """
        self._print(f"\nFetching macro data from {start_date} to {end_date}...")
        
        macro_data = pd.DataFrame()
        
        for name, symbol in self.macro_variables.items():
            try:
                self._print(f"  → {name} ({symbol})...", end='')
                
                # Determine source (FRED or Yahoo Finance)
                if symbol in ['CL=F', 'GC=F', 'DX-Y.NYB', '^VIX']:
                    # Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)['Close']
                    data = data.rename(name)
                    # Remove timezone info to make it timezone-naive
                    if hasattr(data.index, 'tz') and data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                else:
                    # FRED - already timezone-naive
                    data = self.fred.get_series(symbol, start_date, end_date)
                    data = data.rename(name)
                
                macro_data[name] = data
                self._print(f" ✓ ({len(data)} points)")
                
            except Exception as e:
                self._print(f" ✗ Failed: {str(e)[:50]}")
                continue
        
        self._print(f"\n✓ Fetched {len(macro_data.columns)} macro variables")
        return macro_data
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """
        Fetch stock price data.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.Series
            Daily closing prices (timezone-naive)
        """
        self._print(f"\nFetching stock data for {ticker}...")
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)['Close']
        
        # Remove timezone info to make it timezone-naive
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        self._print(f"✓ Fetched {len(data)} daily prices")
        return data
    
    def prepare_data_for_regression(
        self,
        stock_prices: pd.Series,
        macro_data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Prepare data for regression analysis.
        
        Steps:
        1. Convert to daily frequency (forward-fill macro data)
        2. Calculate percentage changes (returns for stock, changes for macro)
        3. Align and drop NaN values
        
        Parameters:
        -----------
        stock_prices : pd.Series
            Stock closing prices
        macro_data : pd.DataFrame
            Macro variables time series
            
        Returns:
        --------
        tuple of (pd.Series, pd.DataFrame)
            (stock_returns, macro_changes) ready for regression
        """
        self._print("\nPreparing data for regression...")
        
        # Align to daily frequency
        stock_daily = stock_prices.resample('D').last().ffill()
        macro_daily = macro_data.resample('D').last().ffill()
        
        # Calculate percentage changes
        stock_returns = stock_daily.pct_change()
        macro_changes = macro_daily.pct_change()
        
        # Align dates
        common_dates = stock_returns.index.intersection(macro_changes.index)
        stock_returns = stock_returns.loc[common_dates]
        macro_changes = macro_changes.loc[common_dates]
        
        # Drop NaN values
        stock_returns = stock_returns.dropna()
        macro_changes = macro_changes.dropna()
        
        # Final alignment
        common_dates = stock_returns.index.intersection(macro_changes.index)
        stock_returns = stock_returns.loc[common_dates]
        macro_changes = macro_changes.loc[common_dates]
        
        self._print(f"✓ Prepared {len(stock_returns)} data points for regression")
        
        return stock_returns, macro_changes
    
    def run_multiple_regression(
        self,
        stock_returns: pd.Series,
        macro_changes: pd.DataFrame
    ) -> Dict:
        """
        Run multiple regression of stock returns on macro changes.
        
        Model: stock_returns = β₀ + β₁*macro₁ + β₂*macro₂ + ... + ε
        
        Parameters:
        -----------
        stock_returns : pd.Series
            Stock returns (dependent variable)
        macro_changes : pd.DataFrame
            Macro variable changes (independent variables)
            
        Returns:
        --------
        Dict
            Regression results including coefficients, p-values, R²
        """
        self._print("\nRunning multiple regression...")
        
        # Align data
        common_idx = stock_returns.index.intersection(macro_changes.index)
        y = stock_returns.loc[common_idx]
        X = macro_changes.loc[common_idx]
        
        # Remove any remaining NaN
        mask = ~(y.isna() | X.isna().any(axis=1))
        y = y[mask]
        X = X[mask]
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # Fit model
        try:
            model = sm.OLS(y, X_with_const).fit()
            
            self._print(f"✓ Regression complete")
            self._print(f"  R-squared: {model.rsquared:.4f}")
            self._print(f"  Observations: {model.nobs:.0f}")
            
            # Extract results
            results = {
                'coefficients': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                't_stats': model.tvalues.to_dict(),
                'model_stats': {
                    'r_squared': float(model.rsquared),
                    'adj_r_squared': float(model.rsquared_adj),
                    'f_statistic': float(model.fvalue),
                    'f_pvalue': float(model.f_pvalue),
                    'n_observations': int(model.nobs)
                },
                'model_object': model  # Keep for further analysis if needed
            }
            
            return results
            
        except Exception as e:
            self._print(f"✗ Regression failed: {e}")
            return {
                'error': str(e),
                'coefficients': {},
                'p_values': {},
                't_stats': {},
                'model_stats': {
                    'r_squared': 0.0,
                    'adj_r_squared': 0.0,
                    'f_statistic': 0.0,
                    'f_pvalue': 1.0,
                    'n_observations': 0
                }
            }
    
    def calculate_sensitivity_scores(
        self,
        regression_results: Dict
    ) -> List[Dict]:
        """
        Calculate sensitivity scores for each macro variable.
        
        Score combines:
        - Absolute coefficient size (effect magnitude)
        - Statistical significance (p-value)
        - Normalized to 0-100 scale
        
        Parameters:
        -----------
        regression_results : Dict
            Output from run_multiple_regression()
            
        Returns:
        --------
        List[Dict]
            Sorted list of sensitivities with scores, betas, p-values
        """
        self._print("\nCalculating sensitivity scores...")
        
        coefficients = regression_results['coefficients']
        p_values = regression_results['p_values']
        
        sensitivities = []
        
        for var_name in coefficients.keys():
            if var_name == 'const':
                continue
            
            beta = coefficients[var_name]
            p_val = p_values[var_name]
            
            # Calculate score
            # Score = |beta| * significance_weight
            # Significance weight: 1.0 if p<0.01, 0.5 if p<0.05, 0.2 if p<0.1, 0.1 otherwise
            if p_val < 0.01:
                sig_weight = 1.0
            elif p_val < 0.05:
                sig_weight = 0.5
            elif p_val < 0.1:
                sig_weight = 0.2
            else:
                sig_weight = 0.1
            
            score = abs(beta) * sig_weight * 1000  # Scale to reasonable range
            
            # Interpretation
            direction = "positive" if beta > 0 else "negative"
            strength = "strong" if p_val < 0.01 else "moderate" if p_val < 0.05 else "weak"
            interpretation = f"Stock has {strength} {direction} sensitivity to {var_name}"
            
            sensitivities.append({
                'variable': var_name,
                'beta': float(beta),
                'p_value': float(p_val),
                'score': float(score),
                'interpretation': interpretation
            })
        
        # Sort by score (highest first)
        sensitivities.sort(key=lambda x: x['score'], reverse=True)
        
        self._print(f"✓ Calculated {len(sensitivities)} sensitivity scores")
        
        return sensitivities
    
    def fetch_news_for_macro_variable(
        self,
        variable_name: str,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Fetch recent news articles related to a macro variable.
        
        Parameters:
        -----------
        variable_name : str
            Name of the macro variable
        top_n : int
            Number of articles to retrieve (default: 5)
            
        Returns:
        --------
        List[Dict]
            List of news articles with title, source, date, url
        """
        if not self.news_api:
            if self.verbose:
                self._print(f"  ⚠ NewsAPI not available for {variable_name}")
            return []
        
        # Get search keywords
        keywords = self.news_keywords.get(variable_name, variable_name)
        
        try:
            # Fetch articles from last 30 days
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            response = self.news_api.get_everything(
                q=keywords,
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=top_n
            )
            
            articles = []
            for article in response.get('articles', [])[:top_n]:
                articles.append({
                    'title': article.get('title', 'N/A'),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', 'N/A')[:10],  # Date only
                    'url': article.get('url', '#'),
                    'description': article.get('description', '')[:200] if article.get('description') else ''
                })
            
            return articles
            
        except Exception as e:
            if self.verbose:
                self._print(f"  ✗ Failed to fetch news for {variable_name}: {e}")
            return []
    
    def analyze(
        self,
        ticker: str,
        lookback_years: int = 3,
        top_drivers: int = 5,
        fetch_news: bool = True
    ) -> Dict:
        """
        Complete macro sensitivity analysis for a stock.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        lookback_years : int
            Years of historical data to analyze (default: 3)
        top_drivers : int
            Number of top macro drivers to report (default: 5)
        fetch_news : bool
            Whether to fetch news for top drivers (default: True)
            
        Returns:
        --------
        Dict
            Complete analysis results including sensitivities and news
            (JSON-serializable)
        """
        if self.verbose:
            self._print("="*70)
            self._print(f"MACRO SENSITIVITY ANALYSIS: {ticker.upper()}")
            self._print("="*70)
        
        # Define date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
        
        self._print(f"\nAnalysis Period: {start_date} to {end_date}")
        self._print(f"Lookback: {lookback_years} years")
        
        # Step 1: Fetch data
        try:
            macro_data = self.fetch_macro_data(start_date, end_date)
            stock_prices = self.fetch_stock_data(ticker, start_date, end_date)
        except Exception as e:
            error_msg = f"Failed to fetch data: {str(e)}"
            self._print(f"✗ {error_msg}")
            return {
                'error': error_msg,
                'ticker': ticker.upper(),
                'analysis_period': f"{start_date} to {end_date}",
                'lookback_years': lookback_years
            }
        
        if macro_data.empty or stock_prices.empty:
            error_msg = "Insufficient data fetched"
            self._print(f"✗ {error_msg}")
            return {
                'error': error_msg,
                'ticker': ticker.upper(),
                'analysis_period': f"{start_date} to {end_date}",
                'lookback_years': lookback_years
            }
        
        # Step 2: Prepare data
        try:
            stock_returns, macro_changes = self.prepare_data_for_regression(
                stock_prices, macro_data
            )
        except Exception as e:
            error_msg = f"Failed to prepare data: {str(e)}"
            self._print(f"✗ {error_msg}")
            return {
                'error': error_msg,
                'ticker': ticker.upper(),
                'analysis_period': f"{start_date} to {end_date}",
                'lookback_years': lookback_years
            }
        
        # Step 3: Run regression
        regression_results = self.run_multiple_regression(stock_returns, macro_changes)
        
        if 'error' in regression_results:
            return {
                'error': regression_results['error'],
                'ticker': ticker.upper(),
                'analysis_period': f"{start_date} to {end_date}",
                'lookback_years': lookback_years
            }
        
        # Step 4: Calculate sensitivities
        sensitivities = self.calculate_sensitivity_scores(regression_results)
        
        # Step 5: Fetch news for top drivers (if enabled)
        if fetch_news and self.news_api:
            if self.verbose:
                self._print(f"\n{'='*70}")
                self._print(f"FETCHING NEWS FOR TOP {top_drivers} DRIVERS")
                self._print(f"{'='*70}")
            
            for i, sensitivity in enumerate(sensitivities[:top_drivers]):
                var_name = sensitivity['variable']
                self._print(f"\n{i+1}. {var_name} (Score: {sensitivity['score']:.1f})")
                
                news = self.fetch_news_for_macro_variable(var_name, top_n=5)
                sensitivity['news'] = news
                
                if news:
                    self._print(f"  ✓ Found {len(news)} articles")
                    if self.verbose:
                        for j, article in enumerate(news, 1):
                            self._print(f"     {j}. {article['title'][:60]}... ({article['source']})")
                else:
                    self._print(f"  ⚠ No news available")
        elif not fetch_news:
            # Add empty news lists if not fetching
            for sensitivity in sensitivities[:top_drivers]:
                sensitivity['news'] = []
        
        # Compile final results (remove model_object for JSON serialization)
        model_stats = regression_results['model_stats'].copy()
        
        results = {
            'ticker': ticker.upper(),
            'analysis_period': f"{start_date} to {end_date}",
            'lookback_years': lookback_years,
            'model_stats': model_stats,
            'top_sensitivities': sensitivities[:top_drivers],
            'all_sensitivities': sensitivities
        }
        
        return results
    
    def format_summary_report(self, results: Dict) -> str:
        """
        Format a summary report of the analysis as a string.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze()
            
        Returns:
        --------
        str
            Formatted summary report
        """
        if not results or 'error' in results:
            return f"Error: {results.get('error', 'Unknown error')}"
        
        lines = []
        lines.append("=" * 70)
        lines.append(f"MACRO SENSITIVITY REPORT: {results['ticker']}")
        lines.append("=" * 70)
        
        # Model statistics
        stats = results['model_stats']
        lines.append(f"\nOVERALL MACRO SENSITIVITY:")
        lines.append(f"  R-squared: {stats['r_squared']:.4f} ({stats['r_squared']*100:.2f}%)")
        lines.append(f"  Interpretation: {stats['r_squared']*100:.1f}% of stock variance explained by macro factors")
        lines.append(f"  Observations: {stats['n_observations']}")
        
        # Top sensitivities
        lines.append(f"\nTOP MACRO DRIVERS:")
        lines.append("-" * 70)
        
        for i, sens in enumerate(results['top_sensitivities'], 1):
            lines.append(f"\n{i}. {sens['variable']}")
            lines.append(f"   Sensitivity Score: {sens['score']:.1f}/100")
            lines.append(f"   Beta Coefficient: {sens['beta']:+.4f}")
            significance = '***' if sens['p_value'] < 0.001 else '**' if sens['p_value'] < 0.01 else '*' if sens['p_value'] < 0.05 else ''
            lines.append(f"   P-value: {sens['p_value']:.4f} {significance}")
            lines.append(f"   Interpretation: {sens['interpretation']}")
            
            # News
            if sens.get('news'):
                lines.append(f"\n   RECENT NEWS ({len(sens['news'])} articles):")
                for j, article in enumerate(sens['news'], 1):
                    lines.append(f"   {j}. [{article['published_at']}] {article['title']}")
                    lines.append(f"      Source: {article['source']}")
                    lines.append(f"      URL: {article['url']}")
                    if j < len(sens['news']):
                        lines.append("")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def print_summary_report(self, results: Dict):
        """
        Print a formatted summary report of the analysis.
        
        Parameters:
        -----------
        results : Dict
            Results from analyze()
        """
        print(self.format_summary_report(results))


def analyze_macro_sensitivity(
    ticker: str,
    fred_api_key: str,
    news_api_key: Optional[str] = None,
    lookback_years: int = 3,
    top_drivers: int = 5,
    fetch_news: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Simple API function to analyze macro sensitivity for a stock.
    
    This is the main API-friendly function that should be used in FastAPI.
    
    Args:
        ticker: Stock ticker symbol
        fred_api_key: FRED API key
        news_api_key: NewsAPI key (optional, for news retrieval)
        lookback_years: Years of historical data (default: 3)
        top_drivers: Number of top drivers to return (default: 5)
        fetch_news: Whether to fetch news (default: True)
        verbose: Print progress messages (default: False)
        
    Returns:
        Dictionary with sensitivity analysis results (JSON-serializable)
        
    Example:
        >>> result = analyze_macro_sensitivity(
        ...     'AAPL',
        ...     fred_api_key='your_key',
        ...     lookback_years=3,
        ...     verbose=False
        ... )
        >>> print(f"R-squared: {result['model_stats']['r_squared']:.2%}")
    """
    analyzer = MacroSensitivityAnalyzer(
        fred_api_key=fred_api_key,
        news_api_key=news_api_key,
        verbose=verbose
    )
    
    return analyzer.analyze(
        ticker=ticker,
        lookback_years=lookback_years,
        top_drivers=top_drivers,
        fetch_news=fetch_news
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage(
    ticker: str = "AAPL",
    fred_api_key: str = None,
    news_api_key: str = None
):
    """Example usage of the Macro Sensitivity Analyzer."""
    
    if not fred_api_key:
        raise ValueError("FRED API key is required")
    
    print(f"Analyzing macro sensitivity for {ticker}...\n")
    
    # Run analysis
    results = analyze_macro_sensitivity(
        ticker=ticker,
        fred_api_key=fred_api_key,
        news_api_key=news_api_key,
        lookback_years=3,
        top_drivers=5,
        fetch_news=True,
        verbose=False
    )
    
    # Print report
    if 'error' not in results:
        analyzer = MacroSensitivityAnalyzer(fred_api_key, news_api_key, verbose=False)
        analyzer.print_summary_report(results)
        
        # Access specific results programmatically
        print("\n\nProgrammatic Access Example:")
        print(f"Overall R-squared: {results['model_stats']['r_squared']:.4f}")
        print(f"Top driver: {results['top_sensitivities'][0]['variable']}")
        print(f"Top driver beta: {results['top_sensitivities'][0]['beta']:.4f}")
    else:
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    
    if not FRED_API_KEY:
        print("Error: FRED_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or set it as an environment variable.")
        print("Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        exit(1)
    
    # Run example
    example_usage(
        ticker="AAPL",
        fred_api_key=FRED_API_KEY,
        news_api_key=NEWS_API_KEY
    )
