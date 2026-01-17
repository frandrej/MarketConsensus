// ============================================================================
// MARKET CONSENSUS - MAIN APPLICATION
// Phase 3: 3D Visualization
// ============================================================================

import * as API from './api.js';
import * as Charts from './charts.js';

// Configuration
const CONFIG = {
    ENABLE_DEBUG: true,
    AUTO_LOAD_DENSITY: true, // Automatically load density when analyzing
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const state = {
    currentTicker: null,
    density: { loading: false, data: null, error: null },
    similar: { loading: false, data: null, error: null },
    sensitivity: { loading: false, data: null, error: null },
    volatility: { loading: false, data: null, error: null },
    activeView: 'density'
};

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const elements = {
    // Form
    tickerForm: document.getElementById('tickerForm'),
    tickerInput: document.getElementById('tickerInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    btnText: document.querySelector('.btn-text'),
    btnLoader: document.querySelector('.btn-loader'),
    
    // Results Section
    resultsSection: document.getElementById('resultsSection'),
    currentTicker: document.getElementById('currentTicker'),
    resetBtn: document.getElementById('resetBtn'),
    
    // Action Buttons
    actionButtons: document.querySelectorAll('.action-btn'),
    
    // View Panels
    densityView: document.getElementById('densityView'),
    similarView: document.getElementById('similarView'),
    sensitivityView: document.getElementById('sensitivityView'),
    volatilityView: document.getElementById('volatilityView'),
    
    // Density elements
    densityLoading: document.getElementById('densityLoading'),
    densityChart: document.getElementById('densityChart'),
    densityError: document.getElementById('densityError'),
    densityErrorMsg: document.getElementById('densityErrorMsg'),
    
    // Similar elements
    similarLoading: document.getElementById('similarLoading'),
    similarResults: document.getElementById('similarResults'),
    similarError: document.getElementById('similarError'),
    similarErrorMsg: document.getElementById('similarErrorMsg'),
    
    // Sensitivity elements
    sensitivityLoading: document.getElementById('sensitivityLoading'),
    sensitivityResults: document.getElementById('sensitivityResults'),
    sensitivityError: document.getElementById('sensitivityError'),
    sensitivityErrorMsg: document.getElementById('sensitivityErrorMsg'),
    
    // Volatility elements
    volatilityLoading: document.getElementById('volatilityLoading'),
    volatilityResults: document.getElementById('volatilityResults'),
    volatilityError: document.getElementById('volatilityError'),
    volatilityErrorMsg: document.getElementById('volatilityErrorMsg'),
    
    // Toast
    toast: document.getElementById('toast'),
    toastIcon: document.getElementById('toastIcon'),
    toastMessage: document.getElementById('toastMessage')
};

// ============================================================================
// INITIALIZATION
// ============================================================================

async function init() {
    console.log('🚀 Market Consensus App Initializing (Phase 3: 3D Visualization)...');
    
    // Setup event listeners
    setupEventListeners();
    
    // Setup chart resize handling
    Charts.setupChartResize('densityChart');
    
    // Check backend health
    await checkBackendHealth();
    
    // Focus on ticker input
    elements.tickerInput.focus();
    
    console.log('✅ App Ready!');
}

// ============================================================================
// BACKEND HEALTH CHECK
// ============================================================================

async function checkBackendHealth() {
    console.log('🏥 Checking backend health...');
    
    const result = await API.checkHealth();
    
    if (result.error) {
        console.warn('⚠️ Backend not available:', result.message);
        showToast('Warning: Backend server not connected. Start backend with: python main.py', 'warning');
    } else {
        console.log('✅ Backend connected:', result);
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Form submission
    elements.tickerForm.addEventListener('submit', handleFormSubmit);
    
    // Reset button
    elements.resetBtn.addEventListener('click', handleReset);
    
    // Action buttons (view switching + data loading)
    elements.actionButtons.forEach(btn => {
        btn.addEventListener('click', () => handleActionButton(btn.dataset.view));
    });
    
    // Uppercase ticker input as user types
    elements.tickerInput.addEventListener('input', (e) => {
        e.target.value = e.target.value.toUpperCase();
    });
}

// ============================================================================
// FORM HANDLING
// ============================================================================

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const ticker = elements.tickerInput.value.trim().toUpperCase();
    
    if (!ticker) {
        showToast('Please enter a ticker symbol', 'error');
        return;
    }
    
    // Validate ticker (basic - just letters, 1-5 chars)
    if (!/^[A-Z]{1,5}$/.test(ticker)) {
        showToast('Invalid ticker format. Please use 1-5 letters.', 'error');
        return;
    }
    
    console.log(`📊 Analyzing ticker: ${ticker}`);
    
    // Show loading on button
    setButtonLoading(true);
    
    // Update state
    state.currentTicker = ticker;
    
    // Show results section
    showResults(ticker);
    
    // Automatically load density data
    if (CONFIG.AUTO_LOAD_DENSITY) {
        await loadDensityData(ticker);
    }
    
    // Hide button loading
    setButtonLoading(false);
    
    showToast(`Loaded data for ${ticker}`, 'success');
}

// ============================================================================
// BUTTON LOADING STATE
// ============================================================================

function setButtonLoading(loading) {
    if (loading) {
        elements.btnText.style.display = 'none';
        elements.btnLoader.style.display = 'inline-block';
        elements.analyzeBtn.disabled = true;
    } else {
        elements.btnText.style.display = 'inline';
        elements.btnLoader.style.display = 'none';
        elements.analyzeBtn.disabled = false;
    }
}

// ============================================================================
// VIEW MANAGEMENT
// ============================================================================

function showResults(ticker) {
    // Update current ticker display
    elements.currentTicker.textContent = ticker;
    
    // Disable ticker input
    elements.tickerInput.disabled = true;
    elements.tickerInput.style.opacity = '0.5';
    elements.tickerInput.style.cursor = 'not-allowed';
    
    // Show results section with animation
    elements.resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function handleReset() {
    console.log('🔄 Resetting...');
    
    // Hide results section
    elements.resultsSection.style.display = 'none';
    
    // Clear and re-enable input
    elements.tickerInput.value = '';
    elements.tickerInput.disabled = false;
    elements.tickerInput.style.opacity = '1';
    elements.tickerInput.style.cursor = 'text';
    
    // Reset state
    state.currentTicker = null;
    state.density = { loading: false, data: null, error: null };
    state.similar = { loading: false, data: null, error: null };
    state.sensitivity = { loading: false, data: null, error: null };
    state.volatility = { loading: false, data: null, error: null };
    state.activeView = 'density';
    
    // Focus on input
    elements.tickerInput.focus();
    
    // Reset active button
    elements.actionButtons.forEach(btn => {
        if (btn.dataset.view === 'density') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Show density view
    showView('density');
    
    showToast('Ready for new search', 'success');
}

async function handleActionButton(viewName) {
    console.log(`🔀 Action button clicked: ${viewName}`);
    
    // Switch view
    handleViewSwitch(viewName);
    
    // Load data if not already loaded
    const ticker = state.currentTicker;
    if (!ticker) return;
    
    switch(viewName) {
        case 'density':
            if (!state.density.data && !state.density.loading) {
                await loadDensityData(ticker);
            }
            break;
        case 'similar':
            if (!state.similar.data && !state.similar.loading) {
                await loadSimilarData(ticker);
            }
            break;
        case 'sensitivity':
            if (!state.sensitivity.data && !state.sensitivity.loading) {
                await loadSensitivityData(ticker);
            }
            break;
        case 'volatility':
            if (!state.volatility.data && !state.volatility.loading) {
                await loadVolatilityData(ticker);
            }
            break;
    }
}

function handleViewSwitch(viewName) {
    console.log(`🔀 Switching to view: ${viewName}`);
    
    state.activeView = viewName;
    
    // Update active button
    elements.actionButtons.forEach(btn => {
        if (btn.dataset.view === viewName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    // Show corresponding view
    showView(viewName);
}

function showView(viewName) {
    // Hide all views
    document.querySelectorAll('.view-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // Show selected view
    const viewMap = {
        'density': elements.densityView,
        'similar': elements.similarView,
        'sensitivity': elements.sensitivityView,
        'volatility': elements.volatilityView
    };
    
    if (viewMap[viewName]) {
        viewMap[viewName].classList.add('active');
    }
}

// ============================================================================
// DATA LOADING: DENSITY
// ============================================================================

async function loadDensityData(ticker) {
    console.log(`📊 Loading density data for ${ticker}...`);
    
    // Update state
    state.density.loading = true;
    state.density.error = null;
    
    // Show loading UI
    elements.densityLoading.style.display = 'flex';
    elements.densityChart.style.display = 'none';
    elements.densityError.style.display = 'none';
    
    // Fetch data
    const result = await API.fetchDensity(ticker, {
        days_forward: 30,
        num_points: 4
    });
    
    // Update state
    state.density.loading = false;
    
    if (result.error) {
        // Show error
        state.density.error = result.message;
        elements.densityLoading.style.display = 'none';
        elements.densityError.style.display = 'flex';
        elements.densityErrorMsg.textContent = result.message;
        showToast(`Failed to load density: ${result.message}`, 'error');
    } else {
        // Show success
        state.density.data = result.data;
        elements.densityLoading.style.display = 'none';
        elements.densityChart.style.display = 'block';
        
        // Render chart (will be implemented in Phase 3)
        renderDensityChart(result.data);
    }
}

function renderDensityChart(data) {
    console.log('📈 Rendering 3D density chart...', data);
    
    // Clear any previous chart
    Charts.clearDensityChart('densityChart');
    
    // Render the 3D surface plot
    Charts.render3DDensity(data, 'densityChart');
}

// ============================================================================
// DATA LOADING: SIMILAR STOCKS
// ============================================================================

async function loadSimilarData(ticker) {
    console.log(`🔍 Loading similar stocks for ${ticker}...`);
    
    state.similar.loading = true;
    state.similar.error = null;
    
    elements.similarLoading.style.display = 'flex';
    elements.similarResults.style.display = 'none';
    elements.similarError.style.display = 'none';
    
    const result = await API.fetchSimilarStocks(ticker, {
        top_n: 3,
        use_fast_filter: true
    });
    
    state.similar.loading = false;
    
    if (result.error) {
        state.similar.error = result.message;
        elements.similarLoading.style.display = 'none';
        elements.similarError.style.display = 'flex';
        elements.similarErrorMsg.textContent = result.message;
        showToast(`Failed to load similar stocks: ${result.message}`, 'error');
    } else {
        state.similar.data = result.data;
        elements.similarLoading.style.display = 'none';
        elements.similarResults.style.display = 'block';
        renderSimilarStocks(result.data);
    }
}

function renderSimilarStocks(data) {
    console.log('🔍 Rendering similar stocks...', data);
    
    const html = `
        <div style="padding: 1rem;">
            <p style="margin-bottom: 1.5rem; color: #64748b;">
                Found ${data.similar_companies.length} similar companies in ${data.target_sector}
            </p>
            ${data.similar_companies.map((company, i) => `
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; border: 2px solid #e2e8f0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="font-size: 1.25rem; font-weight: 600; color: #0f172a;">${i + 1}. ${company.ticker}</h4>
                        <span style="background: #dbeafe; color: #2563eb; padding: 0.25rem 0.75rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">
                            ${(1 - company.distance).toFixed(2)}% similar
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; font-size: 0.875rem; color: #64748b;">
                        <div>Sector: <strong>${company.sector}</strong></div>
                        <div>Price: <strong>$${company.price.toFixed(2)}</strong></div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    elements.similarResults.innerHTML = html;
}

// ============================================================================
// DATA LOADING: SENSITIVITY
// ============================================================================

async function loadSensitivityData(ticker) {
    console.log(`📊 Loading sensitivity analysis for ${ticker}...`);
    
    state.sensitivity.loading = true;
    state.sensitivity.error = null;
    
    elements.sensitivityLoading.style.display = 'flex';
    elements.sensitivityResults.style.display = 'none';
    elements.sensitivityError.style.display = 'none';
    
    const result = await API.fetchSensitivity(ticker, {
        lookback_years: 3,
        include_news: true  // Changed to true to fetch news
    });
    
    state.sensitivity.loading = false;
    
    if (result.error) {
        state.sensitivity.error = result.message;
        elements.sensitivityLoading.style.display = 'none';
        elements.sensitivityError.style.display = 'flex';
        elements.sensitivityErrorMsg.textContent = result.message;
        showToast(`Failed to load sensitivity: ${result.message}`, 'error');
    } else {
        state.sensitivity.data = result.data;
        elements.sensitivityLoading.style.display = 'none';
        elements.sensitivityResults.style.display = 'block';
        renderSensitivity(result.data);
    }
}

function renderSensitivity(data) {
    console.log('📊 Rendering sensitivity...', data);
    
    const html = `
        <div style="padding: 1rem;">
            <h4 style="font-size: 1.125rem; font-weight: 600; margin-bottom: 1.5rem;">Top Macro Drivers</h4>
            
            ${data.top_sensitivities.map((sens, i) => {
                // Create interpretation with beta coefficient
                const ticker = state.currentTicker || 'Stock';
                const factor = sens.variable;
                const beta = sens.beta;
                const direction = beta > 0 ? 'increases' : 'decreases';
                const magnitude = Math.abs(beta * 100).toFixed(2)*0.01;
                
                let betaInterpretation = '';
                if (factor.toLowerCase().includes('rate') || factor.toLowerCase().includes('yield')) {
                    betaInterpretation = `If ${factor} increases by 1 percentage point, ${ticker} ${direction} by ${magnitude}%`;
                } else if (factor.toLowerCase().includes('vix')) {
                    betaInterpretation = `If VIX increases by 1 point, ${ticker} ${direction} by ${magnitude}%`;
                } else if (factor.toLowerCase().includes('inflation') || factor.toLowerCase().includes('cpi')) {
                    betaInterpretation = `If inflation increases by 1 percentage point, ${ticker} ${direction} by ${magnitude}%`;
                } else if (factor.toLowerCase().includes('gdp') || factor.toLowerCase().includes('growth')) {
                    betaInterpretation = `If ${factor} increases by 1%, ${ticker} ${direction} by ${magnitude}%`;
                } else {
                    betaInterpretation = `If ${factor} increases by 1%, ${ticker} ${direction} by ${magnitude}%`;
                }
                
                // Get top 3 news articles if available
                const newsHtml = sens.news && sens.news.length > 0 ? `
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                        <h6 style="font-size: 0.875rem; font-weight: 600; color: #64748b; margin-bottom: 0.75rem;">
                            📰 Recent News About ${sens.variable}
                        </h6>
                        ${sens.news.slice(0, 3).map(article => `
                            <div style="margin-bottom: 0.75rem; padding: 0.75rem; background: #f8fafc; border-radius: 0.375rem; border-left: 3px solid #2563eb;">
                                <a href="${article.url}" target="_blank" rel="noopener noreferrer" style="color: #2563eb; text-decoration: none; font-size: 0.875rem; font-weight: 500; display: block; margin-bottom: 0.25rem; line-height: 1.4;">
                                    ${article.title}
                                </a>
                                <div style="font-size: 0.75rem; color: #64748b;">
                                    <span style="font-weight: 600;">${article.source}</span>
                                    ${article.published_date ? ` • ${new Date(article.published_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : '';
                
                return `
                <div style="margin-bottom: 1.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #e2e8f0;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;">
                        <h5 style="font-size: 1rem; font-weight: 600; color: #0f172a;">${i + 1}. ${sens.variable}</h5>
                        <span style="font-size: 0.875rem; color: #64748b;">
                            β = ${beta.toFixed(4)}
                            ${sens.p_value < 0.01 ? '***' : sens.p_value < 0.05 ? '**' : sens.p_value < 0.1 ? '*' : ''}
                        </span>
                    </div>
                    <p style="color: #0f172a; font-size: 0.95rem; margin-bottom: 0.75rem; font-weight: 500;">
                        ${betaInterpretation}
                    </p>
                    <div style="background: ${beta > 0 ? '#dcfce7' : '#fee2e2'}; height: 8px; border-radius: 4px; position: relative; overflow: hidden;">
                        <div style="background: ${beta > 0 ? '#10b981' : '#ef4444'}; height: 100%; width: ${Math.min(Math.abs(sens.score) * 2, 100)}%;"></div>
                    </div>
                    ${newsHtml}
                </div>
            `;
            }).join('')}
        </div>
    `;
    
    elements.sensitivityResults.innerHTML = html;
}

// ============================================================================
// DATA LOADING: VOLATILITY
// ============================================================================

async function loadVolatilityData(ticker) {
    console.log(`⚡ Loading volatility prediction for ${ticker}...`);
    
    state.volatility.loading = true;
    state.volatility.error = null;
    
    elements.volatilityLoading.style.display = 'flex';
    elements.volatilityResults.style.display = 'none';
    elements.volatilityError.style.display = 'none';
    
    const result = await API.fetchVolatility(ticker);
    
    state.volatility.loading = false;
    
    if (result.error) {
        state.volatility.error = result.message;
        elements.volatilityLoading.style.display = 'none';
        elements.volatilityError.style.display = 'flex';
        elements.volatilityErrorMsg.textContent = result.message;
        showToast(`Failed to load volatility: ${result.message}`, 'error');
    } else {
        state.volatility.data = result.data;
        elements.volatilityLoading.style.display = 'none';
        elements.volatilityResults.style.display = 'block';
        renderVolatility(result.data);
    }
}

function renderVolatility(data) {
    console.log('⚡ Rendering volatility...', data);
    
    const predictionColors = {
        'decrease': { bg: '#dcfce7', text: '#166534', icon: '🟢' },
        'stay_same': { bg: '#fef3c7', text: '#92400e', icon: '🟡' },
        'increase': { bg: '#fee2e2', text: '#991b1b', icon: '🔴' }
    };
    
    const colors = predictionColors[data.prediction] || predictionColors.stay_same;
    
    const html = `
        <div style="padding: 1rem;">
            <div style="background: ${colors.bg}; padding: 2rem; border-radius: 0.75rem; text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">${colors.icon}</div>
                <h3 style="font-size: 1.75rem; font-weight: 700; color: ${colors.text}; text-transform: uppercase; margin-bottom: 0.5rem;">
                    ${data.prediction.replace('_', ' ')}
                </h3>
                <p style="font-size: 1.25rem; color: ${colors.text};">
                    Confidence: ${(data.confidence * 100).toFixed(1)}%
                </p>
            </div>
            
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #0f172a;">
                    Most Important Features for This Prediction
                </h4>
                <p style="font-size: 0.875rem; color: #64748b; margin-bottom: 1.5rem;">
                    These three variables had the highest impact on the model's volatility forecast:
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                    <div style="background: white; padding: 1.25rem; border-radius: 0.5rem; text-align: center; border: 2px solid #e2e8f0;">
                        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase;">VIX</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #2563eb; margin-bottom: 0.25rem;">${data.market_context.vix.toFixed(2)}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Market Fear Index</div>
                    </div>
                    <div style="background: white; padding: 1.25rem; border-radius: 0.5rem; text-align: center; border: 2px solid #e2e8f0;">
                        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase;">SPY</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #2563eb; margin-bottom: 0.25rem;">$${data.market_context.spy.toFixed(2)}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Market Level</div>
                    </div>
                    <div style="background: white; padding: 1.25rem; border-radius: 0.5rem; text-align: center; border: 2px solid #e2e8f0;">
                        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase;">Current IV</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #2563eb; margin-bottom: 0.25rem;">${(data.market_context.current_iv * 100).toFixed(1)}%</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Implied Volatility</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    elements.volatilityResults.innerHTML = html;
}

// ============================================================================
// RETRY FUNCTIONS
// ============================================================================

window.retryDensity = function() {
    console.log('🔄 Retrying density...');
    if (state.currentTicker) {
        loadDensityData(state.currentTicker);
    }
};

window.retrySimilar = function() {
    console.log('🔄 Retrying similar...');
    if (state.currentTicker) {
        loadSimilarData(state.currentTicker);
    }
};

window.retrySensitivity = function() {
    console.log('🔄 Retrying sensitivity...');
    if (state.currentTicker) {
        loadSensitivityData(state.currentTicker);
    }
};

window.retryVolatility = function() {
    console.log('🔄 Retrying volatility...');
    if (state.currentTicker) {
        loadVolatilityData(state.currentTicker);
    }
};

// ============================================================================
// TOAST NOTIFICATIONS
// ============================================================================

function showToast(message, type = 'success') {
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    elements.toastIcon.textContent = icons[type] || icons.info;
    elements.toastMessage.textContent = message;
    
    elements.toast.className = `toast ${type}`;
    elements.toast.classList.add('show');
    
    setTimeout(() => {
        hideToast();
    }, 4000);
}

function hideToast() {
    elements.toast.classList.remove('show');
}

window.hideToast = hideToast;

// ============================================================================
// START APPLICATION
// ============================================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

console.log('📄 app.js loaded (Phase 3 - 3D Visualization)');
