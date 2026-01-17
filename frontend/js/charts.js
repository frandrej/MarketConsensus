// ============================================================================
// CHARTS - 3D DENSITY VISUALIZATION
// Uses Plotly.js for interactive 3D surface plots
// ============================================================================

/**
 * Render 3D Risk-Neutral Density using Plotly
 * @param {Object} data - Density data from API
 * @param {string} containerId - ID of container element
 */
export function render3DDensity(data, containerId) {
    console.log('📈 Rendering 3D density chart with Plotly...', data);
    
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }
    
    // Extract data from API response
    const {
        time_grid,
        strike_grid,
        density_grid,
        metadata
    } = data;
    
    // Prepare the surface plot data
    // Plotly expects z as a 2D array where z[i][j] = density at time_grid[i], strike_grid[j]
    const trace = {
        type: 'surface',
        x: strike_grid,           // Strike prices (X-axis)
        y: time_grid,             // Days to expiration (Y-axis)
        z: density_grid,          // Probability densities (Z-axis)
        colorscale: [
            [0, '#6b21a8'],       // Deep purple (low probability)
            [0.25, '#7c3aed'],    // Purple
            [0.5, '#a78bfa'],     // Medium purple
            [0.75, '#fbbf24'],    // Amber/gold
            [1.0, '#f59e0b']      // Orange-yellow (high probability)
        ],
        colorbar: {
            title: {
                text: 'Probability<br>Density',
                side: 'right'
            },
            thickness: 20,
            len: 0.7,
            x: 1.02
        },
        hovertemplate: 
            '<b>Strike Price:</b> $%{x:.2f}<br>' +
            '<b>Days to Expiration:</b> %{y:.1f}<br>' +
            '<b>Probability Density:</b> %{z:.4f}<br>' +
            '<extra></extra>',
        contours: {
            z: {
                show: true,
                usecolormap: true,
                highlightcolor: "#fff",
                project: { z: true }
            }
        },
        lighting: {
            ambient: 0.8,
            diffuse: 0.8,
            specular: 0.2,
            roughness: 0.5,
            fresnel: 0.2
        }
    };
    
    const plotData = [trace];
    
    // Get current stock price from metadata
    const currentPrice = metadata.underlying[0];
    
    // Layout configuration
    const layout = {
        title: {
            text: `${metadata.ticker} Risk-Neutral Probability Density`,
            font: {
                size: 20,
                family: 'Inter, sans-serif',
                color: '#0f172a'
            }
        },
        scene: {
            xaxis: {
                title: {
                    text: 'Strike Price ($)',
                    font: { size: 14, family: 'Inter, sans-serif' }
                },
                gridcolor: '#cbd5e1',
                showbackground: true,
                backgroundcolor: '#f8fafc'
            },
            yaxis: {
                title: {
                    text: 'Days to Expiration',
                    font: { size: 14, family: 'Inter, sans-serif' }
                },
                gridcolor: '#cbd5e1',
                showbackground: true,
                backgroundcolor: '#f8fafc'
            },
            zaxis: {
                title: {
                    text: 'Probability Density',
                    font: { size: 14, family: 'Inter, sans-serif' }
                },
                gridcolor: '#cbd5e1',
                showbackground: true,
                backgroundcolor: '#f8fafc'
            },
            camera: {
                eye: {
                    x: 1.5,
                    y: 1.5,
                    z: 1.3
                },
                center: {
                    x: 0,
                    y: 0,
                    z: 0
                }
            },
            aspectmode: 'manual',
            aspectratio: {
                x: 1.2,
                y: 1,
                z: 0.7
            }
        },
        autosize: true,
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 60
        },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#f8fafc',
        font: {
            family: 'Inter, sans-serif',
            color: '#64748b'
        }
    };
    
    // Configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: `${metadata.ticker}_density_${new Date().toISOString().split('T')[0]}`,
            height: 1080,
            width: 1920,
            scale: 2
        }
    };
    
    // Create the plot
    Plotly.newPlot(container, plotData, layout, config);
    
    // Add metadata summary below chart
    addMetadataSummary(container, metadata, currentPrice, strike_grid, time_grid);
    
    console.log('✅ 3D density chart rendered successfully');
}

/**
 * Add metadata summary below the chart
 */
function addMetadataSummary(container, metadata, currentPrice, strikeGrid, timeGrid) {
    // Create summary div if it doesn't exist
    let summaryDiv = container.parentElement.querySelector('.density-summary');
    
    if (!summaryDiv) {
        summaryDiv = document.createElement('div');
        summaryDiv.className = 'density-summary';
        container.parentElement.appendChild(summaryDiv);
    }
    
    // Calculate statistics
    const minStrike = Math.min(...strikeGrid);
    const maxStrike = Math.max(...strikeGrid);
    const minDays = Math.min(...timeGrid);
    const maxDays = Math.max(...timeGrid);
    const numExpirations = timeGrid.length;
    const numStrikes = strikeGrid.length;
    
    // Format dates
    const formattedDates = metadata.dates.map(date => {
        const d = new Date(date);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    }).join(', ');
    
    summaryDiv.innerHTML = `
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; margin-top: 1rem;">
            <h4 style="font-size: 1rem; font-weight: 600; color: #0f172a; margin-bottom: 1rem;">
                Density Summary
            </h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem; text-transform: uppercase; font-weight: 600;">
                        Current Price
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #2563eb;">
                        $${currentPrice.toFixed(2)}
                    </div>
                </div>
                
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem; text-transform: uppercase; font-weight: 600;">
                        Strike Range
                    </div>
                    <div style="font-size: 1rem; font-weight: 600; color: #0f172a;">
                        $${minStrike.toFixed(0)} - $${maxStrike.toFixed(0)}
                    </div>
                </div>
                
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem; text-transform: uppercase; font-weight: 600;">
                        Time Range
                    </div>
                    <div style="font-size: 1rem; font-weight: 600; color: #0f172a;">
                        ${minDays.toFixed(0)} - ${maxDays.toFixed(0)} days
                    </div>
                </div>
                
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem; text-transform: uppercase; font-weight: 600;">
                        Data Points
                    </div>
                    <div style="font-size: 1rem; font-weight: 600; color: #0f172a;">
                        ${numExpirations} expirations × ${numStrikes} strikes
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem; text-transform: uppercase; font-weight: 600;">
                    Expiration Dates
                </div>
                <div style="font-size: 0.875rem; color: #0f172a; line-height: 1.6;">
                    ${formattedDates}
                </div>
            </div>
            
            <div style="margin-top: 1rem; padding: 1rem; background: #fef3c7; border-radius: 0.375rem; border-left: 4px solid #f59e0b;">
                <div style="font-size: 0.875rem; color: #78350f; line-height: 1.6;">
                    <strong>💡 How to interpret:</strong> Warmer orange-yellow areas indicate prices the market expects with greater probability. 
                    Purple areas show less probable prices. The surface shows how these expectations evolve across different time horizons.
                </div>
            </div>
        </div>
    `;
}

/**
 * Clear density chart
 */
export function clearDensityChart(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        Plotly.purge(container);
        container.innerHTML = '';
    }
}

/**
 * Resize chart when window resizes
 */
export function setupChartResize(containerId) {
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const container = document.getElementById(containerId);
            if (container && container.data) {
                Plotly.Plots.resize(container);
            }
        }, 250);
    });
}

// ============================================================================
// EXPORT ALL
// ============================================================================

export default {
    render3DDensity,
    clearDensityChart,
    setupChartResize
};
