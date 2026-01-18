"""
Breeden-Litzenberger Risk-Neutral Density Analyzer
==========================================================
- Uses Polygon snapshot options chain.
- Robustly extracts prices from last_quote bid/ask with fallbacks.
- Combines calls/puts using put-call parity (liquidity/open-interest weighting).
- Cleans data using the methodology:
  open interest filter + moneyness filter + gentle monotonicity enforcement.
- Computes risk-neutral density via Breeden-Litzenberger.
- Returns JSON-serializable data for API use (plus optional Plotly visualization).
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from massive import RESTClient
from scipy import interpolate
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Note: yfinance not available")


class BreedenlitzenbergerAnalyzer:
    def __init__(self, api_key: str, verbose: bool = False):
        self.api_key = api_key
        self.client = RESTClient(api_key)
        self.verbose = verbose

        self._treasury_data = None
        self._treasury_tenors = None
        self._treasury_yields = None

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    # =========================================================================
    # DATA ACQUISITION
    # =========================================================================

    def get_underlying_price(self, ticker: str, options_data: Optional[List] = None) -> Optional[float]:
        # Method 1: Yahoo Finance
        if YFINANCE_AVAILABLE:
            try:
                stock = yf.Ticker(ticker)
                price = stock.history(period="1d")["Close"].iloc[-1]
                if price and price > 0:
                    self._print(f"  ✓ Underlying price (Yahoo): ${price:.2f}")
                    return float(price)
            except Exception:
                pass

        # Method 2: Estimate from option data
        if options_data:
            try:
                estimates = []
                for option in options_data[:50]:
                    if not getattr(option, "details", None):
                        continue
                    strike = getattr(option.details, "strike_price", None)
                    if not strike:
                        continue

                    price = None
                    day = getattr(option, "day", None)
                    if day is not None:
                        price = getattr(day, "close", None)

                    if price and price > 10:
                        estimated = price + strike
                        if 50 < estimated < 10000:
                            estimates.append(estimated)

                if estimates:
                    price = float(np.median(estimates))
                    self._print(f"  ✓ Underlying price (estimated): ${price:.2f}")
                    return price
            except Exception:
                pass

        self._print("  ✗ Could not fetch underlying price")
        return None

    def get_risk_free_rate(self, T: float) -> float:
        """
        Treasury yield interpolation/extrapolation. Cached curve, per-T interpolation.
        Returns decimal (e.g., 0.045).
        """
        if self._treasury_data is None:
            try:
                # Fetch most recent treasury curve
                treasury_list = []
                for data_point in self.client.list_treasury_yields(limit=1000, sort="date.desc"):
                    treasury_list.append(data_point)
                    break

                if not treasury_list:
                    self._print("  ⚠ No Treasury data, defaulting to 4.5%")
                    return 0.045

                self._treasury_data = treasury_list[0]
                date_str = getattr(self._treasury_data, "date", "N/A")
                self._print(f"  ✓ Treasury data fetched for {date_str}")

                tenor_map = {
                    1 / 12: "yield_1_month",
                    3 / 12: "yield_3_month",
                    6 / 12: "yield_6_month",
                    1.0: "yield_1_year",
                    2.0: "yield_2_year",
                    3.0: "yield_3_year",
                    5.0: "yield_5_year",
                    7.0: "yield_7_year",
                    10.0: "yield_10_year",
                    20.0: "yield_20_year",
                    30.0: "yield_30_year",
                }

                tenors, yields = [], []
                for tenor_years, field in tenor_map.items():
                    value = getattr(self._treasury_data, field, None)
                    if value is not None:
                        tenors.append(tenor_years)
                        yields.append(value)

                if not tenors:
                    self._print("  ⚠ No valid yields, defaulting to 4.5%")
                    return 0.045

                sort_idx = np.argsort(tenors)
                self._treasury_tenors = np.array(tenors)[sort_idx]
                self._treasury_yields = np.array(yields)[sort_idx]

                if self.verbose:
                    self._print("  ✓ Yield curve available:")
                    for t, y in zip(self._treasury_tenors, self._treasury_yields):
                        label = f"{t*12:.1f} months" if t < 1 else f"{t:.1f} years"
                        self._print(f"     {label}: {y:.3f}%")
            except Exception as e:
                self._print(f"  ⚠ Treasury error: {e}, defaulting to 4.5%")
                return 0.045

        min_tenor = float(self._treasury_tenors[0])
        max_tenor = float(self._treasury_tenors[-1])

        if T < min_tenor:
            # Linear extrapolation short end
            if len(self._treasury_tenors) >= 2:
                t1, t2 = float(self._treasury_tenors[0]), float(self._treasury_tenors[1])
                y1, y2 = float(self._treasury_yields[0]), float(self._treasury_yields[1])
                slope = (y2 - y1) / (t2 - t1)
                rate = y1 + slope * (T - t1)
            else:
                rate = float(self._treasury_yields[0])
        elif T > max_tenor:
            rate = float(self._treasury_yields[-1])
        else:
            rate = float(np.interp(T, self._treasury_tenors, self._treasury_yields))

        return rate / 100.0

    def fetch_both_calls_and_puts(
        self, ticker: str, expiration_date: str, T: float
    ) -> Tuple[List, List, Optional[float], float]:
        self._print("  → Fetching calls...")
        calls = list(
            self.client.list_snapshot_options_chain(
                ticker,
                params={"contract_type": "call", "expiration_date": expiration_date, "limit": 250},
            )
        )

        self._print("  → Fetching puts...")
        puts = list(
            self.client.list_snapshot_options_chain(
                ticker,
                params={"contract_type": "put", "expiration_date": expiration_date, "limit": 250},
            )
        )

        self._print(f"  ✓ Calls: {len(calls)}, Puts: {len(puts)}")

        S0 = self.get_underlying_price(ticker, calls + puts)
        r = self.get_risk_free_rate(T)

        return calls, puts, S0, r

    # =========================================================================
    # ROBUST OPTION PRICE EXTRACTION (Polygon snapshots)
    # =========================================================================

    @staticmethod
    def _get_bid_ask_mid(option) -> Optional[Tuple[float, float, float]]:
        """
        Returns (bid, ask, mid). Uses:
          1) option.last_quote.bid/ask
          2) option.fair_market_value (single price)
          3) option.day.close (single price)
          4) option.last_trade.price (single price)
        """
        bid = ask = None

        lq = getattr(option, "last_quote", None)
        if lq is not None:
            bid = getattr(lq, "bid", None)
            ask = getattr(lq, "ask", None)

        def usable(x):
            return x is not None and np.isfinite(x) and x > 0

        if not (usable(bid) and usable(ask)):
            fmv = getattr(option, "fair_market_value", None)
            if usable(fmv):
                bid = ask = float(fmv)

        if not (usable(bid) and usable(ask)):
            day = getattr(option, "day", None)
            close = getattr(day, "close", None) if day is not None else None
            if usable(close):
                bid = ask = float(close)

        if not (usable(bid) and usable(ask)):
            lt = getattr(option, "last_trade", None)
            price = getattr(lt, "price", None) if lt is not None else None
            if usable(price):
                bid = ask = float(price)

        if not (usable(bid) and usable(ask)):
            return None

        mid = (float(bid) + float(ask)) / 2.0
        return float(bid), float(ask), float(mid)

    def _extract_option_data(self, option) -> Tuple[Optional[float], Optional[float], int]:
        """
        Extract (strike, mid_price, open_interest) from Polygon snapshot option.
        """
        try:
            details = getattr(option, "details", None)
            if not details:
                return None, None, 0

            strike = getattr(details, "strike_price", None)
            if strike is None:
                return None, None, 0

            oi = getattr(option, "open_interest", 0) or 0

            triple = self._get_bid_ask_mid(option)
            if triple is None:
                return float(strike), None, int(oi)

            _, _, mid = triple
            if mid is None or mid <= 0:
                return float(strike), None, int(oi)

            return float(strike), float(mid), int(oi)
        except Exception:
            return None, None, 0

    # =========================================================================
    # PUT-CALL PARITY COMBINATION (old methodology, improved extraction)
    # =========================================================================

    def apply_put_call_parity(
        self,
        calls: List,
        puts: List,
        underlying_price: float,
        r: float,
        T: float,
    ) -> pd.DataFrame:
        """
        Combine calls and puts using put-call parity:
          C - P = S - K*exp(-rT)

        Strategy:
          - If both call and put exist: blend market call with synthetic call from put,
            weighting by open interest (fallback to 50/50 when OI=0).
          - If only call exists: use call.
          - If only put exists: synthesize call from put.

        Returns a DataFrame with:
          strike_price, mid_price, open_interest, underlying_price
        """
        self._print(f"  → Applying put-call parity (r={r*100:.3f}%, T={T:.4f})")
        discount = np.exp(-r * T)

        calls_dict: Dict[float, Dict[str, float]] = {}
        for opt in calls:
            K, price, oi = self._extract_option_data(opt)
            if K is not None and price is not None and price > 0:
                calls_dict[K] = {"price": float(price), "oi": float(oi)}

        puts_dict: Dict[float, Dict[str, float]] = {}
        for opt in puts:
            K, price, oi = self._extract_option_data(opt)
            if K is not None and price is not None and price > 0:
                puts_dict[K] = {"price": float(price), "oi": float(oi)}

        all_strikes = sorted(set(calls_dict.keys()) | set(puts_dict.keys()))
        combined = []

        for K in all_strikes:
            call_data = calls_dict.get(K)
            put_data = puts_dict.get(K)

            if call_data and put_data:
                C_market = call_data["price"]
                P_market = put_data["price"]
                C_synth = P_market + underlying_price - K * discount

                call_oi = call_data["oi"]
                put_oi = put_data["oi"]
                total_oi = call_oi + put_oi

                if total_oi > 0:
                    C_final = (C_market * call_oi + C_synth * put_oi) / total_oi
                    final_oi = int(total_oi)
                else:
                    C_final = 0.5 * (C_market + C_synth)
                    final_oi = 0

            elif call_data:
                C_final = call_data["price"]
                final_oi = int(call_data["oi"])

            elif put_data:
                C_final = put_data["price"] + underlying_price - K * discount
                final_oi = int(put_data["oi"])

            else:
                continue

            if C_final is None or not np.isfinite(C_final) or C_final <= 0:
                continue

            combined.append(
                {
                    "strike_price": float(K),
                    "mid_price": float(C_final),
                    "open_interest": int(final_oi),
                    "underlying_price": float(underlying_price),
                }
            )

        if not combined:
            return pd.DataFrame()

        df = pd.DataFrame(combined).sort_values("strike_price").reset_index(drop=True)
        self._print(f"  ✓ Combined to {len(df)} strike prices")
        return df

    # =========================================================================
    # CLEANING (old methodology; avoids over-pruning)
    # =========================================================================

    def clean_option_data(
        self,
        df: pd.DataFrame,
        min_open_interest: int = 5,
        moneyness_range: Tuple[float, float] = (0.85, 1.15),
    ) -> pd.DataFrame:
        """
        Approach:
          1) Keep positive prices and OI >= threshold
          2) Filter by moneyness K/S
          3) Enforce gentle monotonicity: keep a subsequence of strictly decreasing call prices
             (does not do aggressive local violation removal)
        """
        if df.empty:
            return df

        initial_count = len(df)

        # Basic filters
        df = df[(df["mid_price"] > 0) & (df["open_interest"] >= min_open_interest)].copy()
        if df.empty:
            self._print(f"  ✓ Cleaned: {initial_count} → 0 (failed OI/price filter)")
            return df

        # Moneyness filter
        df["moneyness"] = df["strike_price"] / df["underlying_price"]
        df = df[(df["moneyness"] >= moneyness_range[0]) & (df["moneyness"] <= moneyness_range[1])].copy()
        if df.empty:
            self._print(f"  ✓ Cleaned: {initial_count} → 0 (failed moneyness filter)")
            return df.drop(columns=["moneyness"], errors="ignore")

        df = df.sort_values("strike_price").reset_index(drop=True)

        # Gentle monotonicity: keep a decreasing subsequence
        if len(df) >= 2:
            prices = df["mid_price"].values
            keep = [0]
            for i in range(1, len(prices)):
                if prices[i] < prices[keep[-1]]:
                    keep.append(i)
            df = df.iloc[keep].reset_index(drop=True)

        df = df.drop(columns=["moneyness"], errors="ignore")

        self._print(f"  ✓ Cleaned: {initial_count} → {len(df)} strikes")
        return df

    # =========================================================================
    # INTERPOLATION + DENSITY (stable, smooth derivative approach)
    # =========================================================================

    def interpolate_prices(self, strikes: np.ndarray, prices: np.ndarray, num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        interp = PchipInterpolator(strikes, prices)
        strikes_new = np.linspace(float(strikes.min()), float(strikes.max()), num_points)
        prices_new = interp(strikes_new)

        # Keep prices positive
        prices_new = np.maximum(prices_new, 0.01)

        # Enforce monotonic decrease softly
        for i in range(1, len(prices_new)):
            if prices_new[i] > prices_new[i - 1]:
                prices_new[i] = prices_new[i - 1] * 0.99

        return strikes_new, prices_new

    def calculate_density(
        self,
        strikes: np.ndarray,
        prices: np.ndarray,
        r: float,
        T: float,
        smoothing: float = 1.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Old robust density methodology: smooth prices, compute gradients, smooth derivatives,
        then apply BL formula. Trim tails and normalize.
        """
        prices_smooth = gaussian_filter1d(prices, sigma=smoothing)
        prices_smooth = gaussian_filter1d(prices_smooth, sigma=smoothing * 0.5)

        dh = strikes[1] - strikes[0]
        d1 = np.gradient(prices_smooth, dh)
        d1_smooth = gaussian_filter1d(d1, sigma=smoothing * 0.5)
        d2 = np.gradient(d1_smooth, dh)
        d2_smooth = gaussian_filter1d(d2, sigma=smoothing * 0.3)

        density = np.exp(r * T) * d2_smooth
        density = np.maximum(density, 0)

        # Trim tails (10% each side)
        trim = int(len(density) * 0.1)
        if trim > 0:
            density[:trim] = 0
            density[-trim:] = 0

        integral = np.trapezoid(density, strikes)
        if integral > 0:
            density /= integral

        return strikes, density

    # =========================================================================
    # MAIN WORKFLOW
    # =========================================================================

    def process_single_expiration(self, ticker: str, expiration_date: str, T: float) -> Optional[Dict]:
        self._print("\n" + "=" * 60)
        self._print(f"Processing: {expiration_date} (T={T:.4f} years, {T*365:.1f} days)")
        self._print("=" * 60)

        calls, puts, S0, r = self.fetch_both_calls_and_puts(ticker, expiration_date, T)
        if S0 is None or not calls or not puts:
            self._print("  ✗ Insufficient data")
            return None

        # Combine via put-call parity
        df = self.apply_put_call_parity(calls, puts, S0, r, T)
        if df.empty:
            self._print("  ✗ No data after put-call parity")
            return None

        # Clean using older methodology (less destructive)
        self._print("  → Cleaning data...")
        df_clean = self.clean_option_data(df)
        if len(df_clean) < 5:
            self._print(f"  ✗ Insufficient cleaned strikes: {len(df_clean)}")
            return None

        # Interpolate
        self._print("  → Interpolating prices...")
        strikes_i, prices_i = self.interpolate_prices(
            df_clean["strike_price"].values,
            df_clean["mid_price"].values,
        )

        # Density
        self._print("  → Computing density...")
        strikes_d, density = self.calculate_density(strikes_i, prices_i, r, T)
        self._print("  ✓ Density computed!")

        return {
            "strikes": strikes_d.tolist(),
            "density": density.tolist(),
            "time": float(T * 365.0),
            "rate": float(r),
            "underlying": float(S0),
            "expiration_date": expiration_date,
        }

    def build_evolution(self, ticker: str, days_forward: int = 30, num_points: int = 4) -> Dict:
        if self.verbose:
            self._print("=" * 70)
            self._print("BREEDEN-LITZENBERGER RISK-NEUTRAL DENSITY ANALYZER")
            self._print("=" * 70)

        today = datetime.now()
        end = today + timedelta(days=days_forward)

        self._print(f"\nFinding expirations for {ticker}...")
        expirations = self._get_expirations(ticker, today, end)

        if not expirations:
            self._print("✗ No expirations found")
            return {}

        self._print(f"✓ Found {len(expirations)} expirations")

        # Uniform sampling
        if len(expirations) > num_points:
            idx = np.linspace(0, len(expirations) - 1, num_points, dtype=int)
            expirations = [expirations[i] for i in idx]

        results = []
        for exp_date in expirations:
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            dt = exp_dt - today
            T = dt.total_seconds() / (365.0 * 24 * 60 * 60)
            if T <= 0:
                continue


            out = self.process_single_expiration(ticker, exp_date, T)
            if out:
                results.append(out)

        if not results:
            self._print("\n✗ No valid results")
            return {}

        # Build common grid
        self._print("\n" + "=" * 60)
        self._print("Building 3D density surface...")
        self._print("=" * 60)

        time_grid = [r["time"] for r in results]

        all_strikes = [r["strikes"] for r in results]
        K_min = min(min(s) for s in all_strikes)
        K_max = max(max(s) for s in all_strikes)
        strike_grid = np.linspace(K_min, K_max, 50).tolist()

        density_grid = []
        for rdict in results:
            interp_fn = interpolate.interp1d(
                rdict["strikes"],
                rdict["density"],
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            density_grid.append(interp_fn(strike_grid).tolist())

        self._print(f"✓ Grid: {len(time_grid)} times × {len(strike_grid)} strikes")

        return {
            "time_grid": time_grid,
            "strike_grid": strike_grid,
            "density_grid": density_grid,
            "metadata": {
                "ticker": ticker,
                "dates": [r["expiration_date"] for r in results],
                "rates": [r["rate"] for r in results],
                "underlying": [r["underlying"] for r in results],
            },
        }

    def _get_expirations(self, ticker: str, start: datetime, end: datetime) -> List[str]:
        exps = set()
        for option in self.client.list_snapshot_options_chain(ticker, params={"contract_type": "call", "limit": 250}):
            details = getattr(option, "details", None)
            if not details:
                continue
            exp = getattr(details, "expiration_date", None)
            if not exp:
                continue
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                if start <= exp_dt <= end:
                    exps.add(exp)
            except Exception:
                pass
        return sorted(exps)

    # =========================================================================
    # OPTIONAL VISUALIZATION
    # =========================================================================

    def plot_3d(self, data: Dict, show: bool = True) -> Optional[object]:
        if not data:
            self._print("No data to plot")
            return None

        try:
            import plotly.graph_objects as go
        except ImportError:
            self._print("Plotly not installed. Cannot create visualization.")
            return None

        time_array = np.array(data["time_grid"], dtype=float)
        strike_array = np.array(data["strike_grid"], dtype=float)
        density_array = np.array(data["density_grid"], dtype=float)

        T_grid, K_grid = np.meshgrid(time_array, strike_array)

        fig = go.Figure(
            data=[
                go.Surface(
                    x=K_grid,
                    y=T_grid,
                    z=density_array.T,
                    name="Density",
                )
            ]
        )

        meta = data["metadata"]
        fig.update_layout(
            title=f"Risk-Neutral Density Evolution - {meta['ticker']}",
            scene=dict(
                xaxis_title="Strike Price ($)",
                yaxis_title="Days to Expiration",
                zaxis_title="Probability Density",
            ),
            width=1000,
            height=700,
        )

        if show:
            fig.show()

        return fig


def compute_density(
    ticker: str,
    api_key: str,
    days_forward: int = 30,
    num_points: int = 4,
    verbose: bool = False,
) -> Dict:
    analyzer = BreedenlitzenbergerAnalyzer(api_key, verbose=verbose)
    return analyzer.build_evolution(ticker=ticker, days_forward=days_forward, num_points=num_points)


def main(ticker: str = "AAPL", api_key: Optional[str] = None, verbose: bool = False) -> Dict:
    if not api_key:
        raise ValueError("API key is required. Pass it as a parameter or set POLYGON_API_KEY env var.")
    return compute_density(ticker=ticker, api_key=api_key, verbose=verbose)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("POLYGON_API_KEY")
    if not API_KEY:
        raise SystemExit("POLYGON_API_KEY is not set in environment variables.")

    data = main(ticker="CRM", api_key=API_KEY, verbose=False)
    if data:
        print("\n✓ Analysis complete!")
        # Optional visualization
        try:
            BreedenlitzenbergerAnalyzer(API_KEY, verbose=False).plot_3d(data, show=True)
        except Exception as e:
            print(f"Plot failed: {e}")
    else:
        print("\n✗ Analysis failed")
