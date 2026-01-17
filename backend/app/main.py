from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

# Import our refactored modules
from density import compute_density
from distance import find_similar_stocks
from sensitivity import analyze_macro_sensitivity
from getprediction import predict_volatility, find_model_directory

# Load environment variables from project root
# Go up two levels: backend/app -> backend -> MarketConsensus
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / '.env'

print(f"DEBUG: Loading .env from: {ENV_PATH}")
print(f"DEBUG: .env exists: {ENV_PATH.exists()}")

load_dotenv(ENV_PATH)

# Verify loading
print(f"DEBUG: POLYGON_API_KEY loaded: {os.getenv('POLYGON_API_KEY') is not None}")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Market Consensus API",
    description="API for sto    ck analysis including risk-neutral density, similar stocks, macro sensitivity, and volatility prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",      # Add this for Live Server
    "http://127.0.0.1:5500"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration from environment variables
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
MODEL_DIR = os.getenv('MODEL_DIR')

# Validate required configuration on startup
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup"""
    logger.info("Starting Market Consensus API...")
    
    # Check required API keys
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not found in environment variables")
        raise RuntimeError("POLYGON_API_KEY is required")
    
    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not found in environment variables")
        raise RuntimeError("FRED_API_KEY is required")
    
    # Check model directory (try auto-detection if not set)
    global MODEL_DIR
    if not MODEL_DIR:
        logger.warning("MODEL_DIR not set, attempting auto-detection...")
        MODEL_DIR = find_model_directory()
        if MODEL_DIR:
            logger.info(f"Model directory auto-detected at: {MODEL_DIR}")
        else:
            logger.error("MODEL_DIR not found. Volatility prediction will not work.")
    else:
        logger.info(f"Using MODEL_DIR: {MODEL_DIR}")
    
    # Log optional configurations
    if NEWS_API_KEY:
        logger.info("NEWS_API_KEY configured - news fetching enabled")
    else:
        logger.warning("NEWS_API_KEY not configured - news fetching disabled")
    
    logger.info("Market Consensus API started successfully!")


# ============================================================================
# RESPONSE MODELS (for API documentation)
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    configuration: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "polygon_api": "configured" if POLYGON_API_KEY else "missing",
            "fred_api": "configured" if FRED_API_KEY else "missing",
            "news_api": "configured" if NEWS_API_KEY else "missing",
            "model_dir": MODEL_DIR if MODEL_DIR else "not found"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "polygon_api": "configured" if POLYGON_API_KEY else "missing",
            "fred_api": "configured" if FRED_API_KEY else "missing",
            "news_api": "configured" if NEWS_API_KEY else "missing",
            "model_dir": MODEL_DIR if MODEL_DIR else "not found",
            "endpoints": "4 active (density, similar, sensitivity, volatility)"
        }
    }


# ============================================================================
# ENDPOINT 1: RISK-NEUTRAL DENSITY
# ============================================================================

@app.get("/api/density/{ticker}")
async def get_density(
    ticker: str,
    days_forward: int = Query(default=30, ge=7, le=90, description="Days forward to analyze (7-90)"),
    num_points: int = Query(default=4, ge=1, le=10, description="Number of expiration dates to sample (1-10)")
):
    """
    Get risk-neutral probability density function for a stock.
    
    This endpoint computes the risk-neutral PDF using the Breeden-Litzenberger formula
    from option chain data.
    
    **Parameters:**
    - **ticker**: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    - **days_forward**: Number of days forward to analyze (default: 30, range: 7-90)
    - **num_points**: Number of expiration dates to sample (default: 4, range: 1-10)
    
    **Returns:**
    - Risk-neutral density data across multiple time points
    - Strike prices and probability densities
    - Metadata including dates, rates, and underlying prices
    
    **Example:**
    ```
    GET /api/density/AAPL?days_forward=30&num_points=4
    ```
    """
    logger.info(f"Density request: ticker={ticker}, days_forward={days_forward}, num_points={num_points}")
    
    try:
        result = compute_density(
            ticker=ticker.upper(),
            api_key=POLYGON_API_KEY,
            days_forward=days_forward,
            num_points=num_points,
            verbose=False
        )
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Could not compute density for {ticker}. Ticker may not exist or has insufficient option data."
            )
        
        logger.info(f"Density computed successfully for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"Error computing density for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error computing density: {str(e)}"
        )


# ============================================================================
# ENDPOINT 2: SIMILAR STOCKS
# ============================================================================

@app.get("/api/similar/{ticker}")
async def get_similar_stocks(
    ticker: str,
    top_n: int = Query(default=3, ge=1, le=10, description="Number of similar stocks to return (1-10)"),
    use_fast_filter: bool = Query(default=True, description="Use price-based pre-filtering"),
    max_candidates: int = Query(default=15, ge=5, le=30, description="Maximum candidates to evaluate (5-30)")
):
    """
    Find stocks with similar risk-neutral probability density functions.
    
    This endpoint finds companies in the same sector with the most similar
    risk-neutral PDFs using Wasserstein distance.
    
    **Parameters:**
    - **ticker**: Stock ticker symbol (e.g., NVDA, AMD, INTC)
    - **top_n**: Number of similar stocks to return (default: 3, range: 1-10)
    - **use_fast_filter**: Use price-based pre-filtering to reduce API calls (default: true)
    - **max_candidates**: Maximum number of candidates to evaluate (default: 15, range: 5-30)
    
    **Returns:**
    - List of most similar companies with distance metrics
    - Target stock information (sector, price)
    - Statistics on peers evaluated
    
    **Example:**
    ```
    GET /api/similar/NVDA?top_n=3&use_fast_filter=true
    ```
    
    **Note:** This endpoint may take 1-3 minutes to complete as it computes PDFs
    for multiple stocks.
    """
    logger.info(f"Similar stocks request: ticker={ticker}, top_n={top_n}, fast_filter={use_fast_filter}")
    
    try:
        result = find_similar_stocks(
            target_ticker=ticker.upper(),
            api_key=POLYGON_API_KEY,
            top_n=top_n,
            use_fast_filter=use_fast_filter,
            max_candidates=max_candidates,
            parallel=False,
            verbose=False
        )
        
        if 'error' in result:
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )
        
        logger.info(f"Found {len(result.get('similar_companies', []))} similar stocks for {ticker}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar stocks for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error finding similar stocks: {str(e)}"
        )


# ============================================================================
# ENDPOINT 3: MACRO SENSITIVITY
# ============================================================================

@app.get("/api/sensitivity/{ticker}")
async def get_macro_sensitivity(
    ticker: str,
    lookback_years: int = Query(default=3, ge=1, le=10, description="Years of historical data (1-10)"),
    top_drivers: int = Query(default=5, ge=1, le=10, description="Number of top drivers to return (1-10)"),
    include_news: bool = Query(default=True, description="Include recent news articles")
):
    """
    Analyze stock sensitivity to macroeconomic factors.
    
    This endpoint performs multiple regression analysis to quantify relationships
    between stock returns and changes in macroeconomic variables.
    
    **Parameters:**
    - **ticker**: Stock ticker symbol (e.g., AAPL, JPM, XOM)
    - **lookback_years**: Years of historical data to analyze (default: 3, range: 1-10)
    - **top_drivers**: Number of top macro drivers to return (default: 5, range: 1-10)
    - **include_news**: Whether to fetch recent news for top drivers (default: true)
    
    **Returns:**
    - Model statistics (R-squared, significance)
    - Top macro sensitivities with beta coefficients and p-values
    - Recent news articles for each top driver (if include_news=true)
    
    **Macro Variables Analyzed:**
    - Interest rates (Fed Funds, 10Y Treasury, 2Y Treasury)
    - Inflation (CPI, PCE, PPI)
    - Employment (Unemployment Rate, Jobless Claims)
    - Growth (GDP, Retail Sales, Manufacturing PMI)
    - Market indicators (VIX, Oil, Gold, Dollar Index)
    
    **Example:**
    ```
    GET /api/sensitivity/AAPL?lookback_years=3&include_news=true
    ```
    
    **Note:** This endpoint may take 60-120 seconds to complete.
    """
    logger.info(f"Sensitivity request: ticker={ticker}, lookback_years={lookback_years}, include_news={include_news}")
    
    try:
        result = analyze_macro_sensitivity(
            ticker=ticker.upper(),
            fred_api_key=FRED_API_KEY,
            news_api_key=NEWS_API_KEY if include_news else None,
            lookback_years=lookback_years,
            top_drivers=top_drivers,
            fetch_news=include_news,
            verbose=False
        )
        
        if 'error' in result:
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )
        
        logger.info(f"Sensitivity analysis completed for {ticker}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing sensitivity for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error analyzing sensitivity: {str(e)}"
        )


# ============================================================================
# ENDPOINT 4: VOLATILITY PREDICTION
# ============================================================================

@app.get("/api/volatility/{ticker}")
async def get_volatility_prediction(ticker: str):
    """
    Predict whether a stock's volatility will increase, decrease, or stay the same.
    
    This endpoint uses a machine learning model trained on historical stock data,
    options data, and market indicators to predict future volatility changes.
    
    **Parameters:**
    - **ticker**: Stock ticker symbol (e.g., AAPL, TSLA, GOOGL)
    
    **Returns:**
    - Prediction: 'increase', 'decrease', or 'stay_same'
    - Confidence level (0-1)
    - Probabilities for each outcome
    - Market context (VIX, SPY, current IV)
    - Current stock price
    
    **Example:**
    ```
    GET /api/volatility/AAPL
    ```
    
    **Note:** This endpoint requires trained model files in MODEL_DIR.
    """
    logger.info(f"Volatility prediction request: ticker={ticker}")
    
    # Check if model directory is available
    if not MODEL_DIR:
        raise HTTPException(
            status_code=503,
            detail="Volatility prediction service unavailable. MODEL_DIR not configured."
        )
    
    try:
        result = predict_volatility(
            ticker=ticker.upper(),
            api_key=POLYGON_API_KEY,
            model_dir=MODEL_DIR,
            verbose=False
        )
        
        if 'error' in result:
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )
        
        logger.info(f"Volatility prediction: {ticker} -> {result.get('prediction')}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting volatility for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error predicting volatility: {str(e)}"
        )


# ============================================================================
# ENDPOINT 5: VOLATILITY PREDICTION (SIMPLE)
# ============================================================================

@app.get("/api/volatility/{ticker}/simple")
async def get_volatility_simple(ticker: str):
    """
    Get simple volatility prediction (just the label).
    
    Returns only the prediction label without additional details.
    Useful for quick checks or simple displays.
    
    **Parameters:**
    - **ticker**: Stock ticker symbol
    
    **Returns:**
    - prediction: 'increase', 'decrease', or 'stay_same'
    - ticker: Stock ticker
    - timestamp: Prediction timestamp
    
    **Example:**
    ```
    GET /api/volatility/AAPL/simple
    ```
    """
    logger.info(f"Simple volatility prediction request: ticker={ticker}")
    
    if not MODEL_DIR:
        raise HTTPException(
            status_code=503,
            detail="Volatility prediction service unavailable. MODEL_DIR not configured."
        )
    
    try:
        result = predict_volatility(
            ticker=ticker.upper(),
            api_key=POLYGON_API_KEY,
            model_dir=MODEL_DIR,
            verbose=False
        )
        
        if 'error' in result:
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )
        
        # Return simplified response
        simple_result = {
            "ticker": result['ticker'],
            "prediction": result['prediction'],
            "timestamp": result['timestamp']
        }
        
        logger.info(f"Simple prediction: {ticker} -> {result.get('prediction')}")
        return simple_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simple volatility prediction for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested endpoint does not exist",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    # For development: reload=True enables auto-reload on code changes
    # For production: remove reload=True and use a production ASGI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        log_level="info"
    )