// ============================================================================
// API COMMUNICATION LAYER
// Handles all HTTP requests to FastAPI backend
// ============================================================================

// API Configuration
const API_CONFIG = {
    BASE_URL: 'http://localhost:8000/api',
    TIMEOUT: 180000, // 3 minutes (for slow endpoints)
    RETRY_ATTEMPTS: 1
};

// ============================================================================
// UTILITY: FETCH WITH TIMEOUT
// ============================================================================

async function fetchWithTimeout(url, options = {}, timeout = API_CONFIG.TIMEOUT) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - please try again');
        }
        throw error;
    }
}

// ============================================================================
// UTILITY: ERROR HANDLER
// ============================================================================

function handleApiError(error, endpoint) {
    console.error(`API Error [${endpoint}]:`, error);
    
    // Network errors
    if (error.message === 'Failed to fetch') {
        return {
            error: true,
            message: 'Cannot connect to backend. Make sure the server is running on port 8000.',
            details: 'Connection failed'
        };
    }
    
    // Timeout errors
    if (error.message.includes('timeout')) {
        return {
            error: true,
            message: 'Request took too long. The server might be busy - please try again.',
            details: 'Timeout'
        };
    }
    
    // Generic errors
    return {
        error: true,
        message: error.message || 'An unexpected error occurred',
        details: error.toString()
    };
}

// ============================================================================
// API: HEALTH CHECK
// ============================================================================

export async function checkHealth() {
    try {
        const response = await fetchWithTimeout(
            `${API_CONFIG.BASE_URL.replace('/api', '')}/health`,
            {},
            5000 // 5 second timeout for health check
        );
        
        if (!response.ok) {
            throw new Error(`Health check failed: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        return handleApiError(error, 'health');
    }
}

// ============================================================================
// API: DENSITY (Risk-Neutral PDF)
// ============================================================================

export async function fetchDensity(ticker, options = {}) {
    const {
        days_forward = 30,
        num_points = 4
    } = options;
    
    console.log(`📊 Fetching density for ${ticker}...`);
    
    try {
        const url = `${API_CONFIG.BASE_URL}/density/${ticker}?days_forward=${days_forward}&num_points=${num_points}`;
        
        const response = await fetchWithTimeout(url, {}, 120000); // 2 minute timeout
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Density data received:', data);
        
        return {
            error: false,
            data: data
        };
        
    } catch (error) {
        return handleApiError(error, 'density');
    }
}

// ============================================================================
// API: SIMILAR STOCKS
// ============================================================================

export async function fetchSimilarStocks(ticker, options = {}) {
    const {
        top_n = 3,
        use_fast_filter = true,
        max_candidates = 15
    } = options;
    
    console.log(`🔍 Fetching similar stocks for ${ticker}...`);
    
    try {
        const url = `${API_CONFIG.BASE_URL}/similar/${ticker}?top_n=${top_n}&use_fast_filter=${use_fast_filter}&max_candidates=${max_candidates}`;
        
        const response = await fetchWithTimeout(url, {}, 240000); // 4 minute timeout
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Similar stocks data received:', data);
        
        return {
            error: false,
            data: data
        };
        
    } catch (error) {
        return handleApiError(error, 'similar');
    }
}

// ============================================================================
// API: MACRO SENSITIVITY
// ============================================================================

export async function fetchSensitivity(ticker, options = {}) {
    const {
        lookback_years = 3,
        top_drivers = 5,
        include_news = false // Default to false for faster response
    } = options;
    
    console.log(`📊 Fetching sensitivity analysis for ${ticker}...`);
    
    try {
        const url = `${API_CONFIG.BASE_URL}/sensitivity/${ticker}?lookback_years=${lookback_years}&top_drivers=${top_drivers}&include_news=${include_news}`;
        
        const response = await fetchWithTimeout(url, {}, 180000); // 3 minute timeout
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Sensitivity data received:', data);
        
        return {
            error: false,
            data: data
        };
        
    } catch (error) {
        return handleApiError(error, 'sensitivity');
    }
}

// ============================================================================
// API: VOLATILITY PREDICTION
// ============================================================================

export async function fetchVolatility(ticker) {
    console.log(`⚡ Fetching volatility prediction for ${ticker}...`);
    
    try {
        const url = `${API_CONFIG.BASE_URL}/volatility/${ticker}`;
        
        const response = await fetchWithTimeout(url, {}, 30000); // 30 second timeout
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Volatility data received:', data);
        
        return {
            error: false,
            data: data
        };
        
    } catch (error) {
        return handleApiError(error, 'volatility');
    }
}

// ============================================================================
// EXPORT ALL
// ============================================================================

export default {
    checkHealth,
    fetchDensity,
    fetchSimilarStocks,
    fetchSensitivity,
    fetchVolatility
};
