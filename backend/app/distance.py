"""
Similar Risk-Neutral PDF Finder - API Version
==============================================

Finds companies in the same sector with the most similar risk-neutral 
probability density functions using the Wasserstein distance metric.

Changes from original:
- Accepts ticker directly and computes target PDF internally
- Made verbose output optional
- API key passed as parameter
- Returns JSON-serializable data
- Removed hardcoded API keys
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class SimilarPDFFinder:
    """
    Find companies with similar risk-neutral PDFs in the same sector.
    
    Uses:
    1. S&P 500 constituents for sector classification
    2. Price-based fast filtering to reduce API calls
    3. Wasserstein distance for PDF similarity
    4. Parallel processing for speed
    """
    
    def __init__(self, analyzer, verbose: bool = False):
        """
        Initialize with a BreedenlitzenbergerAnalyzer instance.
        
        Parameters:
        -----------
        analyzer : BreedenlitzenbergerAnalyzer
            The analyzer instance to use for computing PDFs
        verbose : bool
            Whether to print progress messages (default: True)
        """
        self.analyzer = analyzer
        self.sp500_data = None
        self.verbose = verbose
        
    def _print(self, message: str, end: str = '\n'):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message, end=end)
    
    def load_sector_data(self):
        """
        Load S&P 500 constituents with sector information.
        
        Data source: GitHub datasets repository (public, free)
        Contains: Symbol, Name, Sector
        """
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            self.sp500_data = pd.read_csv(url)
            
            # Check what columns are actually available
            self._print(f"✓ Loaded {len(self.sp500_data)} S&P 500 companies")
            if self.verbose:
                self._print(f"  Available columns: {list(self.sp500_data.columns)}")
            
            # Handle different possible column names
            # Common variations: 'Sector', 'GICS Sector', 'sector'
            sector_column = None
            for col in self.sp500_data.columns:
                if 'sector' in col.lower():
                    sector_column = col
                    break
            
            if sector_column and sector_column != 'Sector':
                # Rename to standard 'Sector'
                self.sp500_data = self.sp500_data.rename(columns={sector_column: 'Sector'})
                if self.verbose:
                    self._print(f"  Renamed column '{sector_column}' → 'Sector'")
            
            # Standardize Symbol column too
            symbol_column = None
            for col in self.sp500_data.columns:
                if col.lower() in ['symbol', 'ticker']:
                    symbol_column = col
                    break
            
            if symbol_column and symbol_column != 'Symbol':
                self.sp500_data = self.sp500_data.rename(columns={symbol_column: 'Symbol'})
                if self.verbose:
                    self._print(f"  Renamed column '{symbol_column}' → 'Symbol'")
            
            # Verify we have the required columns
            if 'Sector' not in self.sp500_data.columns or 'Symbol' not in self.sp500_data.columns:
                raise ValueError("Missing required columns after renaming")
                
        except Exception as e:
            self._print(f"✗ Failed to load S&P 500 data: {e}")
            self._print("  Using fallback sector mapping...")
            # Fallback: comprehensive sector mapping for major stocks
            self.sp500_data = pd.DataFrame({
                'Symbol': [
                    # Technology
                    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'ORCL', 
                    'CSCO', 'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM', 'TXN', 'AMAT',
                    # Consumer Discretionary
                    'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX',
                    # Financials
                    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',
                    # Healthcare
                    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT',
                    # Communication Services
                    'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T',
                    # Consumer Staples
                    'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO',
                    # Industrials
                    'BA', 'HON', 'UPS', 'CAT', 'RTX', 'LMT', 'GE',
                    # Energy
                    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',
                    # Utilities
                    'NEE', 'DUK', 'SO', 'D', 'AEP',
                ],
                'Sector': (
                    ['Technology'] * 16 +
                    ['Consumer Discretionary'] * 8 +
                    ['Financials'] * 8 +
                    ['Health Care'] * 8 +
                    ['Communication Services'] * 7 +
                    ['Consumer Staples'] * 7 +
                    ['Industrials'] * 7 +
                    ['Energy'] * 6 +
                    ['Utilities'] * 5
                )
            })
    
    def get_sector_peers(self, ticker: str) -> Tuple[List[str], str]:
        """
        Get list of peer companies in the same sector.
        
        Parameters:
        -----------
        ticker : str
            Target ticker symbol
            
        Returns:
        --------
        tuple of (List[str], str)
            (list of peer tickers, sector name)
        """
        if self.sp500_data is None:
            self.load_sector_data()
        
        # Find target company's sector
        target_rows = self.sp500_data[self.sp500_data['Symbol'] == ticker.upper()]
        
        if target_rows.empty:
            # If not in S&P 500, try to get sector from Yahoo Finance
            try:
                stock = yf.Ticker(ticker)
                sector = stock.info.get('sector', 'Unknown')
                self._print(f"  ⚠ {ticker} not in S&P 500, sector from yfinance: {sector}")
                return [], sector
            except:
                self._print(f"  ✗ Could not determine sector for {ticker}")
                return [], "Unknown"
        
        sector = target_rows['Sector'].iloc[0]
        
        # Get all companies in same sector
        peers = self.sp500_data[
            self.sp500_data['Sector'] == sector
        ]['Symbol'].tolist()
        
        return peers, sector
    
    def fast_filter_by_price(
        self,
        target_ticker: str,
        target_price: float,
        peers: List[str],
        top_k: int = 15
    ) -> List[str]:
        """
        Fast pre-filter using stock price similarity.
        
        Rationale: Companies with similar prices often have similar option 
        chains and PDF shapes. This reduces API calls dramatically.
        
        Parameters:
        -----------
        target_ticker : str
            Target ticker
        target_price : float
            Current price of target stock
        peers : List[str]
            List of peer tickers
        top_k : int
            Number of candidates to keep (default: 15)
            
        Returns:
        --------
        List[str]
            Filtered list of peer tickers
        """
        self._print(f"\n  → Fast filtering by price similarity...")
        
        price_scores = []
        
        for peer in peers:
            if peer.upper() == target_ticker.upper():
                continue
            
            try:
                stock = yf.Ticker(peer)
                history = stock.history(period='1d')
                
                if history.empty:
                    continue
                
                peer_price = history['Close'].iloc[-1]
                
                # Calculate price ratio
                price_ratio = peer_price / target_price
                
                # Prefer stocks within 30%-300% of target price
                if 0.3 < price_ratio < 3.0:
                    # Score by how close to 1.0 the ratio is
                    score = abs(np.log(price_ratio))
                    price_scores.append((peer, score, peer_price))
                
            except Exception as e:
                continue
        
        # Sort by score (lower is better)
        price_scores.sort(key=lambda x: x[1])
        
        # Take top K
        filtered = [ticker for ticker, score, price in price_scores[:top_k]]
        
        self._print(f"  ✓ Filtered to {len(filtered)} candidates")
        if self.verbose and len(filtered) > 0:
            self._print(f"  Top 5 by price similarity:")
            for ticker, score, price in price_scores[:5]:
                self._print(f"    {ticker}: ${price:.2f} (score: {score:.3f})")
        
        return filtered
    
    def compute_pdf_for_ticker(
        self,
        ticker: str,
        days_forward: int = 30,
        num_points: int = 1
    ) -> Optional[Dict]:
        """
        Compute PDF for a single ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        days_forward : int
            Days forward to analyze (default: 30)
        num_points : int
            Number of expiration points (default: 1)
            
        Returns:
        --------
        Dict or None
            PDF data dictionary or None if computation fails
        """
        try:
            # Set analyzer to non-verbose to reduce noise
            original_verbose = self.analyzer.verbose
            self.analyzer.verbose = False
            
            result = self.analyzer.build_evolution(
                ticker=ticker,
                days_forward=days_forward,
                num_points=num_points
            )
            
            # Restore original verbose setting
            self.analyzer.verbose = original_verbose
            
            if result and 'density_grid' in result and len(result['density_grid']) > 0:
                self._print(f"  ✓ {ticker}")
                return result
            else:
                self._print(f"  ✗ {ticker} (no data)")
                return None
                
        except Exception as e:
            self._print(f"  ✗ {ticker} error: {str(e)[:50]}")
            return None
    
    def calculate_wasserstein_distance(
        self,
        pdf_data1: Dict,
        pdf_data2: Dict,
        normalize: bool = True
    ) -> float:
        """
        Calculate Wasserstein distance between two PDFs.
        
        Parameters:
        -----------
        pdf_data1 : Dict
            First PDF data (from build_evolution)
        pdf_data2 : Dict
            Second PDF data (from build_evolution)
        normalize : bool
            Whether to normalize to [0, 1] range (default: True)
            
        Returns:
        --------
        float
            Wasserstein distance between the two PDFs
        """
        # Extract first time point from each PDF
        strikes1 = np.array(pdf_data1['strike_grid'])
        strikes2 = np.array(pdf_data2['strike_grid'])
        
        # Get density at first time point
        density1 = np.array(pdf_data1['density_grid'][0])
        density2 = np.array(pdf_data2['density_grid'][0])
        
        # Create common strike grid
        min_strike = max(strikes1.min(), strikes2.min())
        max_strike = min(strikes1.max(), strikes2.max())
        common_strikes = np.linspace(min_strike, max_strike, 100)
        
        # Interpolate both densities to common grid
        d1_interp = np.interp(common_strikes, strikes1, density1)
        d2_interp = np.interp(common_strikes, strikes2, density2)
        
        # Renormalize after interpolation
        if np.trapezoid(d1_interp, common_strikes) > 0:
            d1_interp = d1_interp / np.trapezoid(d1_interp, common_strikes)
        if np.trapezoid(d2_interp, common_strikes) > 0:
            d2_interp = d2_interp / np.trapezoid(d2_interp, common_strikes)

        # Calculate Wasserstein distance
        distance = wasserstein_distance(
            common_strikes, common_strikes,
            d1_interp, d2_interp
        )
        
        return distance
    
    def find_similar_pdfs(
        self,
        target_ticker: str,
        top_n: int = 3,
        use_fast_filter: bool = True,
        max_candidates: int = 15,
        parallel: bool = False,
        days_forward: int = 30,
        num_points: int = 1
    ) -> Dict:
        """
        Find companies with most similar risk-neutral PDFs.
        
        This is the main API method that accepts a ticker and automatically
        computes the target PDF internally.
        
        Workflow:
        1. Compute PDF for target ticker
        2. Get sector peers from S&P 500
        3. Fast filter by price similarity (optional)
        4. Compute PDFs for candidates (parallel optional)
        5. Calculate Wasserstein distances
        6. Return top N most similar
        
        Parameters:
        -----------
        target_ticker : str
            Target ticker symbol
        top_n : int
            Number of similar companies to return (default: 3)
        use_fast_filter : bool
            Use price-based filtering (default: True)
        max_candidates : int
            Maximum candidates to evaluate (default: 15)
        parallel : bool
            Use parallel processing (default: False)
        days_forward : int
            Days forward for PDF calculation (default: 30)
        num_points : int
            Number of time points for PDF (default: 1)
            
        Returns:
        --------
        Dict
            Dictionary with:
            - target_ticker: str
            - target_sector: str
            - target_price: float
            - similar_companies: List[Dict] with ticker, distance, sector, price
            - total_peers_in_sector: int
            - candidates_evaluated: int
        """
        if self.verbose:
            self._print("\n" + "="*70)
            self._print(f"FINDING SIMILAR PDFs FOR {target_ticker.upper()}")
            self._print("="*70)
        
        # Step 1: Compute PDF for target ticker
        self._print(f"\nStep 1: Computing PDF for {target_ticker}...")
        target_pdf_data = self.compute_pdf_for_ticker(
            target_ticker, 
            days_forward=days_forward,
            num_points=num_points
        )
        
        if not target_pdf_data:
            error_msg = f"Failed to compute PDF for {target_ticker}"
            self._print(f"✗ {error_msg}")
            return {
                'error': error_msg,
                'target_ticker': target_ticker,
                'similar_companies': []
            }
        
        self._print(f"✓ Target PDF computed successfully")
        
        # Step 2: Get sector peers
        self._print(f"\nStep 2: Finding sector peers...")
        peers, sector = self.get_sector_peers(target_ticker)
        
        if not peers:
            self._print(f"✗ No peers found for {target_ticker}")
            return {
                'target_ticker': target_ticker,
                'target_sector': sector,
                'target_price': target_pdf_data['metadata']['underlying'][0],
                'similar_companies': [],
                'total_peers_in_sector': 0,
                'candidates_evaluated': 0,
                'error': 'No peers found in sector'
            }
        
        self._print(f"✓ Sector: {sector}")
        self._print(f"✓ Total peers in sector: {len(peers)}")
        
        # Step 3: Fast filter (optional)
        target_price = target_pdf_data['metadata']['underlying'][0]
        
        if use_fast_filter:
            self._print(f"\nStep 3: Fast filtering by price...")
            candidates = self.fast_filter_by_price(
                target_ticker, target_price, peers, top_k=max_candidates
            )
        else:
            candidates = [p for p in peers if p.upper() != target_ticker.upper()]
            candidates = candidates[:max_candidates]
            self._print(f"\nStep 3: Skipping fast filter, using first {len(candidates)} peers")
        
        if not candidates:
            self._print("✗ No valid candidates after filtering")
            return {
                'target_ticker': target_ticker,
                'target_sector': sector,
                'target_price': target_price,
                'similar_companies': [],
                'total_peers_in_sector': len(peers),
                'candidates_evaluated': 0,
                'error': 'No valid candidates after filtering'
            }
        
        # Step 4: Compute PDFs for all candidates
        self._print(f"\nStep 4: Computing PDFs for {len(candidates)} candidates...")
        if self.verbose:
            self._print(f"{'='*70}")
        
        candidate_pdfs = {}
        
        if parallel:
            # Parallel computation
            self._print("  Using parallel processing...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ticker = {
                    executor.submit(
                        self.compute_pdf_for_ticker, 
                        ticker, 
                        days_forward=days_forward, 
                        num_points=num_points
                    ): ticker
                    for ticker in candidates
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        if result:
                            candidate_pdfs[ticker] = result
                    except Exception as e:
                        self._print(f"  ✗ {ticker} exception: {e}")
        else:
            # Sequential computation
            for ticker in candidates:
                result = self.compute_pdf_for_ticker(
                    ticker,
                    days_forward=days_forward,
                    num_points=num_points
                )
                if result:
                    candidate_pdfs[ticker] = result
        
        self._print(f"\n✓ Successfully computed {len(candidate_pdfs)} PDFs")
        
        if not candidate_pdfs:
            return {
                'target_ticker': target_ticker,
                'target_sector': sector,
                'target_price': target_price,
                'similar_companies': [],
                'total_peers_in_sector': len(peers),
                'candidates_evaluated': len(candidates),
                'error': 'Could not compute PDFs for any candidates'
            }
        
        # Step 5: Calculate similarities
        self._print(f"\nStep 5: Calculating Wasserstein distances...")
        if self.verbose:
            self._print(f"{'='*70}")
        
        similarities = []
        
        for ticker, pdf_data in candidate_pdfs.items():
            try:
                distance = self.calculate_wasserstein_distance(
                    target_pdf_data, pdf_data, normalize=True
                )
                
                peer_price = pdf_data['metadata']['underlying'][0]
                
                similarities.append({
                    'ticker': ticker,
                    'distance': float(distance),
                    'sector': sector,
                    'price': float(peer_price)
                })
                
                self._print(f"  {ticker}: distance = {distance:.6f}, price = ${peer_price:.2f}")
                
            except Exception as e:
                self._print(f"  ✗ {ticker} similarity error: {e}")
                continue
        
        # Step 6: Sort and return top N
        similarities.sort(key=lambda x: x['distance'])
        top_matches = similarities[:top_n]
        
        if self.verbose:
            self._print(f"\n{'='*70}")
            self._print(f"TOP {top_n} MOST SIMILAR COMPANIES")
            self._print(f"{'='*70}")
            
            for i, match in enumerate(top_matches, 1):
                self._print(f"\n{i}. {match['ticker']}")
                self._print(f"   Wasserstein Distance: {match['distance']:.6f}")
                self._print(f"   Sector: {match['sector']}")
                self._print(f"   Current Price: ${match['price']:.2f}")
        
        # Return structured result
        return {
            'target_ticker': target_ticker,
            'target_sector': sector,
            'target_price': float(target_price),
            'similar_companies': top_matches,
            'total_peers_in_sector': len(peers),
            'candidates_evaluated': len(candidates),
            'pdfs_computed': len(candidate_pdfs)
        }


def find_similar_stocks(
    target_ticker: str,
    api_key: str,
    top_n: int = 3,
    use_fast_filter: bool = True,
    max_candidates: int = 15,
    parallel: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Simple API function to find stocks with similar risk-neutral PDFs.
    
    This is the main API-friendly function that should be used in FastAPI.
    
    Args:
        target_ticker: Stock ticker to find similar companies for
        api_key: Polygon.io API key
        top_n: Number of similar stocks to return (default: 3)
        use_fast_filter: Use price-based pre-filtering (default: True)
        max_candidates: Maximum candidates to evaluate (default: 15)
        parallel: Use parallel processing (default: False)
        verbose: Print progress messages (default: False)
        
    Returns:
        Dictionary with similar companies and metadata (JSON-serializable)
        
    Example:
        >>> result = find_similar_stocks('AAPL', api_key='your_key', top_n=3)
        >>> for company in result['similar_companies']:
        ...     print(f"{company['ticker']}: distance={company['distance']:.4f}")
    """
    # Import here to avoid circular imports
    from density import BreedenlitzenbergerAnalyzer
    
    # Create analyzer
    analyzer = BreedenlitzenbergerAnalyzer(api_key, verbose=verbose)
    
    # Create finder
    finder = SimilarPDFFinder(analyzer, verbose=verbose)
    
    # Find similar PDFs
    result = finder.find_similar_pdfs(
        target_ticker=target_ticker,
        top_n=top_n,
        use_fast_filter=use_fast_filter,
        max_candidates=max_candidates,
        parallel=parallel
    )
    
    return result


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage(api_key: str, ticker: str = "AAPL"):
    """
    Example of how to use the Similar PDF Finder with the new API.
    """
    print(f"Finding similar stocks to {ticker}...\n")
    
    result = find_similar_stocks(
        target_ticker=ticker,
        api_key=api_key,
        top_n=3,
        use_fast_filter=True,
        parallel=False,
        verbose=False
    )
    
    # Display results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if 'error' in result:
        print(f"\nError: {result['error']}")
    elif result['similar_companies']:
        print(f"\nTarget: {result['target_ticker']}")
        print(f"Sector: {result['target_sector']}")
        print(f"Price: ${result['target_price']:.2f}")
        print(f"\nMost similar companies:")
        for i, company in enumerate(result['similar_companies'], 1):
            print(f"{i}. {company['ticker']} - distance: {company['distance']:.6f}, price: ${company['price']:.2f}")
        print(f"\nStatistics:")
        print(f"  Total peers in sector: {result['total_peers_in_sector']}")
        print(f"  Candidates evaluated: {result['candidates_evaluated']}")
        print(f"  PDFs computed: {result['pdfs_computed']}")
    else:
        print("No similar companies found")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    API_KEY = os.getenv('POLYGON_API_KEY')
    
    if not API_KEY:
        print("Error: POLYGON_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key or set it as an environment variable.")
        exit(1)
    
    # Run example
    example_usage(API_KEY, ticker="AAPL")