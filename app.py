"""
COMPREHENSIVE PORTFOLIO OPTIMIZATION PLATFORM
Multi-Asset, Multi-Region, Interactive Portfolio Construction & Optimization
"""

# ============================================
# IMPORTS
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt import plotting as pf_plotting
import warnings
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from scipy import stats
from scipy.optimize import minimize
import itertools

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Global Portfolio Optimizer Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ASSET UNIVERSE DEFINITION
# ============================================

# Pre-defined asset universe
ASSET_UNIVERSE = {
    "US_STOCKS": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp",
        "GOOGL": "Alphabet Inc (Class A)",
        "AMZN": "Amazon.com Inc",
        "META": "Meta Platforms Inc",
        "TSLA": "Tesla Inc",
        "NVDA": "NVIDIA Corp",
        "JPM": "JPMorgan Chase & Co",
        "JNJ": "Johnson & Johnson",
        "V": "Visa Inc",
        "PG": "Procter & Gamble Co",
        "UNH": "UnitedHealth Group Inc",
        "HD": "Home Depot Inc",
        "MA": "Mastercard Inc",
        "DIS": "Walt Disney Co",
        "BRK-B": "Berkshire Hathaway Inc",
        "XOM": "Exxon Mobil Corp",
        "BAC": "Bank of America Corp",
        "PFE": "Pfizer Inc",
        "CSCO": "Cisco Systems Inc"
    },
    
    "US_ETFS": {
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "DIA": "SPDR Dow Jones Industrial Average ETF",
        "IWM": "iShares Russell 2000 ETF",
        "VTI": "Vanguard Total Stock Market ETF",
        "VOO": "Vanguard S&P 500 ETF",
        "IVV": "iShares Core S&P 500 ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "SHY": "iShares 1-3 Year Treasury Bond ETF",
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "VNQ": "Vanguard Real Estate ETF",
        "XLK": "Technology Select Sector SPDR Fund",
        "XLE": "Energy Select Sector SPDR Fund",
        "XLF": "Financial Select Sector SPDR Fund",
        "XLV": "Health Care Select Sector SPDR Fund"
    },
    
    "CRYPTO": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana",
        "ADA-USD": "Cardano",
        "DOT-USD": "Polkadot",
        "DOGE-USD": "Dogecoin",
        "MATIC-USD": "Polygon",
        "AVAX-USD": "Avalanche",
        "LINK-USD": "Chainlink",
        "UNI-USD": "Uniswap"
    },
    
    "COMMODITIES": {
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "CL=F": "Crude Oil Futures",
        "NG=F": "Natural Gas Futures",
        "ZC=F": "Corn Futures",
        "ZS=F": "Soybean Futures",
        "ZW=F": "Wheat Futures",
        "HG=F": "Copper Futures"
    },
    
    "TURKISH_STOCKS": {
        "THYAO.IS": "Turk Hava Yollari",
        "KCHOL.IS": "Koc Holding",
        "SASA.IS": "Sasa Polyester",
        "ASELS.IS": "Aselsan",
        "AKBNK.IS": "Akbank",
        "ARCLK.IS": "Arcelik",
        "BIMAS.IS": "BIM Birlesik Magazalar",
        "EKGYO.IS": "Emlak Konut Gayrimenkul",
        "EREGL.IS": "Eregli Demir Celik",
        "FROTO.IS": "Ford Otosan",
        "GARAN.IS": "Garanti BBVA",
        "HALKB.IS": "Turkiye Halk Bankasi",
        "ISCTR.IS": "Turkiye Is Bankasi",
        "KRDMD.IS": "Kardemir Karabuk Demir Celik",
        "KOZAA.IS": "Koza Anadolu Metal Madencilik",
        "KOZAL.IS": "Koza Altin Isletmeleri",
        "PETKM.IS": "Petkim Petrokimya Holding",
        "PGSUS.IS": "Pegasus Hava Tasimaciligi",
        "SAHOL.IS": "Haci Omer Sabanci Holding",
        "SISE.IS": "Turkiye Sise Cam Fabrikalari",
        "TCELL.IS": "Turkcell Iletisim Hizmetleri",
        "TKFEN.IS": "Tekfen Holding",
        "TOASO.IS": "Tofas Turk Otomobil Fabrikasi",
        "TTKOM.IS": "Turk Telekomunikasyon",
        "TUPRS.IS": "Tupras Turkiye Petrol Rafinerileri",
        "ULKER.IS": "Ulker Biskuvi Sanayi",
        "VAKBN.IS": "Turkiye Vakiflar Bankasi",
        "YATAS.IS": "Yatas Yatak ve Yorgan Sanayi",
        "YKBNK.IS": "Yapi ve Kredi Bankasi",
        "ZOREN.IS": "Zorlu Enerji Elektrik Uretim"
    },
    
    "TURKISH_ETFS": {
        "XU100.IS": "BIST 100 Index",
        "XGIDA.IS": "BIST Food Index",
        "XUSIN.IS": "BIST Industrial Index",
        "XUTEK.IS": "BIST Technology Index"
    }
}

# ============================================
# RISK-FREE RATE FUNCTIONS
# ============================================

def get_treasury_yield_treasury_direct():
    """Fetch from U.S. Treasury Direct (official source)"""
    try:
        url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/XmlView.aspx?data=yield"
        response = requests.get(url, timeout=10)
        
        # Parse XML
        import re
        # Look for 10-year yield
        match = re.search(r'<d:BC_10YEAR[^>]*>([\d.]+)</d:BC_10YEAR>', response.text)
        if match:
            yield_value = float(match.group(1))
            return yield_value, "U.S. Treasury Direct"
    except:
        pass
    return None, None

def get_treasury_yield_yahoo():
    """Fetch Treasury yield from Yahoo Finance"""
    try:
        # Try ^TNX first
        tnx = yf.download("^TNX", period="5d", progress=False)
        if not tnx.empty and 'Close' in tnx.columns:
            yield_value = tnx['Close'].iloc[-1]
            return yield_value, "Yahoo Finance (^TNX)"
        
        # Try alternative
        tlt = yf.download("TLT", period="5d", progress=False)
        if not tlt.empty and 'Close' in tlt.columns:
            # Approximate yield from TLT price (very rough)
            yield_value = 4.0  # Default approximation
            return yield_value, "Yahoo Finance (TLT approx)"
    except:
        pass
    return None, None

def get_risk_free_rate(fred_api_key=None, region="US"):
    """
    Get risk-free rate for specified region
    
    Args:
        region: "US", "TR" (Turkey), or "CUSTOM"
    """
    if region == "TR":
        # Turkey uses higher risk-free rate
        return 0.20, "Turkey Central Bank (20% default)"
    
    elif region == "US":
        # Try multiple sources for US
        sources = [
            ("U.S. Treasury Direct", get_treasury_yield_treasury_direct),
            ("Yahoo Finance", get_treasury_yield_yahoo)
        ]
        
        # Try FRED if API key provided
        if fred_api_key:
            def get_fred_yield():
                try:
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": "DGS10",
                        "api_key": fred_api_key,
                        "file_type": "json",
                        "observation_start": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                        "sort_order": "desc",
                        "limit": 1
                    }
                    response = requests.get(url, params=params, timeout=5)
                    data = response.json()
                    if "observations" in data and data["observations"]:
                        yield_value = float(data["observations"][0]["value"])
                        return yield_value, "FRED API"
                except:
                    return None, None
            sources.insert(0, ("FRED API", get_fred_yield))
        
        for source_name, source_func in sources:
            result, source = source_func()
            if result is not None:
                return result / 100, source  # Convert to decimal
        
        # Default fallback
        return 0.04, "Default (4%)"
    
    else:
        # Custom region
        return 0.03, "Custom (3% default)"

# ============================================
# DATA LOADING AND PROCESSING
# ============================================

@st.cache_data(ttl=3600)
def load_asset_data(asset_list, period="1y", progress_callback=None):
    """Load multiple assets with caching"""
    data_dict = {}
    failed_assets = []
    
    for i, (ticker, name) in enumerate(asset_list.items()):
        if progress_callback:
            progress_callback(i, len(asset_list), f"Loading {ticker}...")
        
        try:
            # Download data
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            
            if data.empty:
                failed_assets.append(ticker)
                continue
            
            # Use Adjusted Close if available, otherwise Close
            if 'Adj Close' in data.columns:
                price_series = data['Adj Close']
            elif 'Close' in data.columns:
                price_series = data['Close']
            else:
                failed_assets.append(ticker)
                continue
            
            # Handle multiple columns (some tickers return OHLC)
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
            
            data_dict[ticker] = price_series
            
        except Exception as e:
            failed_assets.append(ticker)
            continue
    
    if data_dict:
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Fill missing values
        df = df.ffill().bfill()
        
        # Remove columns with all NaN
        df = df.dropna(axis=1, how='all')
        
        # Calculate returns
        returns_df = df.pct_change().dropna()
        
        return returns_df, df, failed_assets
    
    return None, None, failed_assets

def calculate_comprehensive_metrics(returns_df, prices_df, risk_free_rate):
    """Calculate comprehensive performance and risk metrics"""
    metrics = {}
    
    for asset in returns_df.columns:
        returns = returns_df[asset]
        prices = prices_df[asset]
        
        # Basic metrics
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Risk metrics
        # VaR (Historical)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        
        # CVaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sortino Ratio (downside risk)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Omega Ratio
        threshold = risk_free_rate / 252  # Daily threshold
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        omega_ratio = gains / losses if losses != 0 else 0
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Gain/Loss Ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        gain_loss_ratio = abs(positive_returns.mean() / negative_returns.mean()) if len(negative_returns) > 0 and negative_returns.mean() != 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt((drawdown ** 2).mean())
        
        # Tail Ratio (95%/5%)
        tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
        
        metrics[asset] = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Omega Ratio': omega_ratio,
            'Max Drawdown': max_drawdown,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'VaR (99%)': var_99,
            'CVaR (99%)': cvar_99,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Gain/Loss Ratio': gain_loss_ratio,
            'Ulcer Index': ulcer_index,
            'Tail Ratio': tail_ratio,
            'Positive Months': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        }
    
    return pd.DataFrame(metrics).T

# ============================================
# PORTFOLIO OPTIMIZATION ENGINE
# ============================================

class GlobalPortfolioOptimizer:
    """Enhanced portfolio optimizer with multiple strategies"""
    
    def __init__(self, returns_df, prices_df, risk_free_rate=0.04):
        self.returns_df = returns_df
        self.prices_df = prices_df
        self.risk_free_rate = risk_free_rate
        self.mu = expected_returns.mean_historical_return(returns_df)
        self.S = risk_models.CovarianceShrinkage(returns_df).ledoit_wolf()
        self.assets = returns_df.columns.tolist()
        
    def optimize_equal_weight(self):
        """Equal weight portfolio"""
        n_assets = len(self.assets)
        weights = {asset: 1/n_assets for asset in self.assets}
        performance = self.calculate_portfolio_performance(weights)
        return weights, performance, "Equal Weight"
    
    def optimize_max_sharpe(self):
        """Maximize Sharpe Ratio"""
        ef = EfficientFrontier(self.mu, self.S)
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)
        return weights, performance, "Max Sharpe"
    
    def optimize_min_volatility(self):
        """Minimize Volatility"""
        ef = EfficientFrontier(self.mu, self.S)
        ef.min_volatility()
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)
        return weights, performance, "Min Volatility"
    
    def optimize_max_return(self, target_volatility=None):
        """Maximize Return with optional volatility target"""
        ef = EfficientFrontier(self.mu, self.S)
        
        if target_volatility:
            try:
                ef.efficient_risk(target_volatility)
            except:
                ef.max_return()
        else:
            ef.max_return()
        
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)
        return weights, performance, "Max Return"
    
    def optimize_max_diversification(self):
        """Maximum Diversification Ratio"""
        # Calculate volatilities
        volatilities = self.returns_df.std() * np.sqrt(252)
        
        # Define diversification ratio objective
        def diversification_ratio(weights):
            w = np.array(weights)
            portfolio_vol = np.sqrt(w @ self.S @ w)
            weighted_vol = w @ volatilities.values
            return -portfolio_vol / weighted_vol  # Negative for minimization
        
        # Optimization
        n_assets = len(self.assets)
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            diversification_ratio,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if result.success:
            weights = {asset: w for asset, w in zip(self.assets, result.x)}
            # Clean small weights
            weights = {k: v for k, v in weights.items() if v > 0.001}
            # Renormalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            performance = self.calculate_portfolio_performance(weights)
            return weights, performance, "Max Diversification"
        else:
            return self.optimize_equal_weight()
    
    def optimize_risk_parity(self):
        """Risk Parity (Equal Risk Contribution)"""
        n_assets = len(self.assets)
        
        def risk_parity_objective(weights):
            w = np.array(weights)
            portfolio_vol = np.sqrt(w @ self.S @ w)
            
            # Calculate risk contributions
            risk_contributions = (w * (self.S @ w)) / portfolio_vol
            
            # Calculate deviation from equal contribution
            target_contribution = 1 / n_assets
            deviations = risk_contributions - target_contribution
            
            return np.sum(deviations ** 2)
        
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            risk_parity_objective,
            initial_weights,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if result.success:
            weights = {asset: w for asset, w in zip(self.assets, result.x)}
            weights = {k: v for k, v in weights.items() if v > 0.001}
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            performance = self.calculate_portfolio_performance(weights)
            return weights, performance, "Risk Parity"
        else:
            return self.optimize_equal_weight()
    
    def optimize_custom_constraints(self, max_weight=0.3, min_weight=0.0):
        """Optimize with custom weight constraints"""
        ef = EfficientFrontier(self.mu, self.S)
        
        # Add constraints
        if max_weight < 1.0:
            ef.add_constraint(lambda w: w <= max_weight)
        if min_weight > 0.0:
            ef.add_constraint(lambda w: w >= min_weight)
        
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)
        
        return weights, performance, f"Constrained (max {max_weight:.0%})"
    
    def calculate_portfolio_performance(self, weights):
        """Calculate portfolio performance metrics"""
        # Ensure weights match assets
        weight_array = np.array([weights.get(asset, 0) for asset in self.assets])
        
        # Portfolio returns
        portfolio_returns = self.returns_df.dot(weight_array)
        
        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        return (annual_return, annual_volatility, sharpe_ratio)
    
    def generate_efficient_frontier(self, points=100):
        """Generate efficient frontier"""
        ef = EfficientFrontier(self.mu, self.S)
        
        # Get min volatility
        ef.min_volatility()
        min_vol = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)[1]
        
        # Get max sharpe
        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        max_vol = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)[1]
        
        target_volatilities = np.linspace(min_vol, max_vol * 1.5, points)
        
        frontier_data = []
        for target_vol in target_volatilities:
            try:
                ef = EfficientFrontier(self.mu, self.S)
                ef.efficient_risk(target_vol)
                perf = ef.portfolio_performance(risk_free_rate=self.risk_free_rate, verbose=False)
                frontier_data.append({
                    'volatility': perf[1],
                    'return': perf[0],
                    'sharpe': perf[2]
                })
            except:
                continue
        
        return pd.DataFrame(frontier_data)
    
    def monte_carlo_simulation(self, n_portfolios=10000):
        """Monte Carlo simulation of random portfolios"""
        np.random.seed(42)
        n_assets = len(self.assets)
        
        results = []
        for _ in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights /= weights.sum()
            
            # Calculate metrics
            portfolio_return = np.dot(weights, self.mu)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            results.append({
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': sharpe,
                'weights': weights
            })
        
        return pd.DataFrame(results)

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_interactive_frontier_chart(frontier_df, monte_carlo_df, 
                                     optimal_points=None, current_weights=None):
    """Create interactive efficient frontier chart"""
    fig = go.Figure()
    
    # Add Monte Carlo points
    fig.add_trace(go.Scatter(
        x=monte_carlo_df['volatility'],
        y=monte_carlo_df['return'],
        mode='markers',
        name='Random Portfolios',
        marker=dict(
            size=4,
            color=monte_carlo_df['sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
            opacity=0.3
        ),
        hovertemplate='<b>Random Portfolio</b><br>' +
                     'Risk: %{x:.2%}<br>' +
                     'Return: %{y:.2%}<br>' +
                     'Sharpe: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Add efficient frontier
    fig.add_trace(go.Scatter(
        x=frontier_df['volatility'],
        y=frontier_df['return'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='darkblue', width=3),
        hovertemplate='<b>Efficient Frontier</b><br>' +
                     'Risk: %{x:.2%}<br>' +
                     'Return: %{y:.2%}<extra></extra>'
    ))
    
    # Add optimal points
    if optimal_points:
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (label, point) in enumerate(optimal_points.items()):
            fig.add_trace(go.Scatter(
                x=[point['volatility']],
                y=[point['return']],
                mode='markers+text',
                name=label,
                marker=dict(
                    size=15,
                    color=colors[i % len(colors)],
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                text=[label],
                textposition="top center",
                hovertemplate=f'<b>{label}</b><br>' +
                             f'Risk: %{{x:.2%}}<br>' +
                             f'Return: %{{y:.2%}}<br>' +
                             f'Sharpe: {point.get("sharpe", 0):.2f}<extra></extra>'
            ))
    
    # Add current portfolio if provided
    if current_weights:
        # Calculate current portfolio metrics
        optimizer = GlobalPortfolioOptimizer(optimal_points[list(optimal_points.keys())[0]]['returns_df'], 
                                           optimal_points[list(optimal_points.keys())[0]]['prices_df'])
        performance = optimizer.calculate_portfolio_performance(current_weights)
        
        fig.add_trace(go.Scatter(
            x=[performance[1]],
            y=[performance[0]],
            mode='markers+text',
            name='Current Portfolio',
            marker=dict(
                size=15,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            text=['Current'],
            textposition="top center",
            hovertemplate='<b>Current Portfolio</b><br>' +
                         f'Risk: %{{x:.2%}}<br>' +
                         f'Return: %{{y:.2%}}<br>' +
                         f'Sharpe: {performance[2]:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Efficient Frontier with Optimization Results',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Annual Volatility (Risk)',
        yaxis_title='Annual Expected Return',
        hovermode='closest',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    fig.update_xaxes(tickformat='.0%')
    fig.update_yaxes(tickformat='.0%')
    
    return fig

def create_allocation_sunburst(weights, asset_categories):
    """Create sunburst chart for portfolio allocation by category"""
    # Prepare data for sunburst
    data = []
    
    for asset, weight in weights.items():
        # Find category
        category = "Other"
        for cat_name, cat_assets in asset_categories.items():
            if asset in cat_assets:
                category = cat_name
                break
        
        data.append({
            'ids': f'{category}/{asset}',
            'labels': asset,
            'parents': category,
            'values': weight,
            'text': f'{asset}<br>{weight:.2%}'
        })
    
    # Add category totals
    categories = {}
    for entry in data:
        category = entry['parents']
        if category not in categories:
            categories[category] = 0
        categories[category] += entry['values']
    
    for category, total in categories.items():
        data.append({
            'ids': category,
            'labels': category,
            'parents': '',
            'values': total,
            'text': f'{category}<br>{total:.2%}'
        })
    
    # Create sunburst
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in data],
        labels=[d['labels'] for d in data],
        parents=[d['parents'] for d in data],
        values=[d['values'] for d in data],
        branchvalues='total',
        text=[d['text'] for d in data],
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>',
        maxdepth=2
    ))
    
    fig.update_layout(
        title={
            'text': 'Portfolio Allocation by Asset Class',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600
    )
    
    return fig

def create_performance_comparison_chart(returns_df, strategies_weights, prices_df):
    """Compare performance of different strategies"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, (strategy_name, weights) in enumerate(strategies_weights.items()):
        # Calculate portfolio returns
        weight_array = np.array([weights.get(asset, 0) for asset in returns_df.columns])
        portfolio_returns = returns_df.dot(weight_array)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Add to chart
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name=strategy_name,
            line=dict(width=2, color=colors[i % len(colors)]),
            hovertemplate=f'<b>{strategy_name}</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: %{y:.2f}<extra></extra>'
        ))
    
    # Add benchmark (equal weight)
    n_assets = len(returns_df.columns)
    equal_weights = {asset: 1/n_assets for asset in returns_df.columns}
    weight_array = np.array([equal_weights.get(asset, 0) for asset in returns_df.columns])
    benchmark_returns = returns_df.dot(weight_array)
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    fig.add_trace(go.Scatter(
        x=benchmark_cumulative.index,
        y=benchmark_cumulative.values,
        mode='lines',
        name='Equal Weight Benchmark',
        line=dict(width=3, color='black', dash='dash'),
        hovertemplate='<b>Equal Weight Benchmark</b><br>' +
                     'Date: %{x}<br>' +
                     'Value: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Strategy Performance Comparison',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='Cumulative Return (Indexed to 1)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig

def create_risk_metrics_chart(risk_metrics_by_category):
    """Create grouped bar chart for risk metrics by category"""
    if not risk_metrics_by_category:
        return go.Figure()
    
    # Prepare data
    categories = list(risk_metrics_by_category.keys())
    metrics = ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = []
        for category in categories:
            if category in risk_metrics_by_category and metric in risk_metrics_by_category[category]:
                values.append(risk_metrics_by_category[category][metric])
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=metric,
            x=categories,
            y=values,
            text=[f'{v:.2%}' if v != 0 else '' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title={
            'text': 'Risk Metrics by Asset Class',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Asset Class',
        yaxis_title='Value',
        yaxis_tickformat='.0%',
        barmode='group',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_drawdown_chart(returns_df, weights):
    """Create drawdown chart for portfolio"""
    # Calculate portfolio returns
    weight_array = np.array([weights.get(asset, 0) for asset in returns_df.columns])
    portfolio_returns = returns_df.dot(weight_array)
    
    # Calculate drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,  # Convert to percentage
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=2),
        name='Drawdown',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Add annotation for maximum drawdown
    fig.add_annotation(
        x=max_dd_date,
        y=max_dd * 100,
        text=f"Max Drawdown: {max_dd*100:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=50,
        ay=-50,
        bgcolor="white",
        bordercolor="red",
        borderwidth=1
    )
    
    fig.update_layout(
        title={
            'text': f'Portfolio Drawdown (Maximum: {max_dd*100:.2f}%)',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x',
        template='plotly_white',
        height=400
    )
    
    return fig

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    # Title and header
    st.title("🌍 Global Multi-Asset Portfolio Optimizer")
    st.markdown("""
    **Build, Analyze, and Optimize Portfolios Across Multiple Asset Classes**
    
    Features:
    - 📊 **Multi-Region Assets**: US Stocks, Turkish Stocks, ETFs, Crypto, Commodities
    - 🎯 **Multiple Optimization Strategies**: Max Sharpe, Min Volatility, Risk Parity, etc.
    - 📈 **Interactive Portfolio Construction**: Drag & adjust weights
    - ⚠️ **Comprehensive Risk Analysis**: VaR, CVaR, Max Drawdown by asset class
    - 🔄 **Real-time Performance Comparison**: Compare multiple strategies
    """)
    
    # Initialize session state
    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = {}
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = []
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Region selection
        region = st.selectbox(
            "Select Region/Portfolio Type:",
            ["Global Diversified", "US Focus", "Turkish Market", "Crypto Portfolio", "Custom"]
        )
        
        # Risk-free rate configuration
        st.subheader("💰 Risk-Free Rate")
        
        rf_source = st.selectbox(
            "Risk-Free Rate Source:",
            ["Auto Detect (US Treasury)", "Custom", "Turkey (20%)"]
        )
        
        custom_rf = None
        fred_api_key = None
        
        if rf_source == "Auto Detect (US Treasury)":
            fred_api_key = st.text_input(
                "FRED API Key (optional for better data):",
                type="password",
                help="Get free API key from https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        elif rf_source == "Custom":
            custom_rf = st.number_input(
                "Custom Risk-Free Rate (%):",
                min_value=0.0,
                max_value=50.0,
                value=4.0,
                step=0.1
            ) / 100
        
        # Data period
        period = st.selectbox(
            "Historical Data Period:",
            ["3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
        
        # Asset selection
        st.subheader("📊 Asset Selection")
        
        # Pre-built portfolios
        portfolio_preset = st.selectbox(
            "Portfolio Preset:",
            [
                "All Assets (Diversified)",
                "US Stocks + ETFs",
                "Turkish Market",
                "Crypto + Commodities",
                "Custom Selection"
            ]
        )
        
        # Asset class selection
        st.write("**Select Asset Classes:**")
        
        selected_classes = {}
        for class_name, assets in ASSET_UNIVERSE.items():
            if st.checkbox(f"{class_name.replace('_', ' ')} ({len(assets)} assets)", value=True):
                selected_classes[class_name] = assets
        
        # Number of assets per class
        max_assets_per_class = st.slider(
            "Max assets per class:",
            min_value=1,
            max_value=30,
            value=10
        )
        
        # Filter assets
        all_assets = {}
        for class_name, assets in selected_classes.items():
            # Take top N assets by market cap (simplified - in reality would need market cap data)
            selected = dict(list(assets.items())[:max_assets_per_class])
            all_assets.update(selected)
        
        # Display selected assets count
        st.info(f"Selected {len(all_assets)} assets for analysis")
        
        # Optimization strategies
        st.subheader("🎯 Optimization Strategies")
        
        strategies = {
            "Equal Weight": st.checkbox("Equal Weight", value=True),
            "Max Sharpe Ratio": st.checkbox("Max Sharpe Ratio", value=True),
            "Min Volatility": st.checkbox("Min Volatility", value=True),
            "Max Diversification": st.checkbox("Max Diversification", value=True),
            "Risk Parity": st.checkbox("Risk Parity", value=True),
            "Max Return": st.checkbox("Max Return", value=False)
        }
        
        # Constraints for constrained optimization
        st.subheader("🔒 Portfolio Constraints")
        
        use_constraints = st.checkbox("Apply weight constraints", value=False)
        
        if use_constraints:
            max_weight = st.slider(
                "Maximum weight per asset (%):",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            ) / 100
            
            min_weight = st.slider(
                "Minimum weight per asset (%):",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            ) / 100
        
        # Action buttons
        st.subheader("🚀 Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            load_data_btn = st.button("📥 Load Data", use_container_width=True)
        
        with col2:
            run_optimization_btn = st.button("🎯 Run Optimization", use_container_width=True)
        
        if load_data_btn:
            st.session_state.data_loaded = False
            st.rerun()
    
    # Main content area
    if not all_assets:
        st.warning("Please select at least one asset class in the sidebar.")
        return
    
    # Load data
    if load_data_btn or 'returns_df' not in st.session_state:
        with st.spinner(f"Loading data for {len(all_assets)} assets..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress_bar.progress(current / total)
                status_text.text(message)
            
            returns_df, prices_df, failed_assets = load_asset_data(
                all_assets, 
                period,
                progress_callback=update_progress
            )
            
            progress_bar.empty()
            status_text.empty()
            
            if returns_df is None or returns_df.empty:
                st.error(f"Failed to load data. {len(failed_assets)} assets failed: {', '.join(failed_assets[:10])}")
                return
            
            # Store in session state
            st.session_state.returns_df = returns_df
            st.session_state.prices_df = prices_df
            st.session_state.all_assets = all_assets
            st.session_state.failed_assets = failed_assets
            st.session_state.data_loaded = True
            
            st.success(f"✅ Successfully loaded {len(returns_df.columns)} assets with {len(returns_df)} data points")
    
    if 'returns_df' not in st.session_state:
        st.info("Click 'Load Data' to begin analysis.")
        return
    
    # Get data from session state
    returns_df = st.session_state.returns_df
    prices_df = st.session_state.prices_df
    all_assets = st.session_state.all_assets
    
    # Get risk-free rate
    if rf_source == "Turkey (20%)":
        risk_free_rate = 0.20
        rf_source_name = "Turkey Central Bank"
    elif rf_source == "Custom" and custom_rf is not None:
        risk_free_rate = custom_rf
        rf_source_name = "Custom Input"
    else:
        risk_free_rate, rf_source_name = get_risk_free_rate(fred_api_key, "US")
    
    # Display risk-free rate
    st.subheader("💰 Risk-Free Rate")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk-Free Rate", f"{risk_free_rate*100:.2f}%")
    
    with col2:
        st.metric("Source", rf_source_name)
    
    with col3:
        st.metric("Period", period)
    
    # Asset statistics
    st.subheader("📊 Asset Statistics")
    
    # Calculate metrics
    metrics_df = calculate_comprehensive_metrics(returns_df, prices_df, risk_free_rate)
    
    # Display in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "⚠️ Risk Metrics", "📋 All Metrics"])
    
    with tab1:
        # Top performers
        st.write("**Top 5 by Sharpe Ratio:**")
        top_sharpe = metrics_df.nlargest(5, 'Sharpe Ratio')[['Annual Return', 'Annual Volatility', 'Sharpe Ratio']]
        st.dataframe(
            top_sharpe.style.format({
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}'
            }),
            use_container_width=True
        )
    
    with tab2:
        # Risk metrics
        risk_cols = ['Max Drawdown', 'VaR (95%)', 'CVaR (95%)', 'Sortino Ratio']
        risk_df = metrics_df[risk_cols]
        st.dataframe(
            risk_df.style.format({
                'Max Drawdown': '{:.2%}',
                'VaR (95%)': '{:.2%}',
                'CVaR (95%)': '{:.2%}',
                'Sortino Ratio': '{:.2f}'
            }),
            use_container_width=True
        )
    
    with tab3:
        # All metrics
        st.dataframe(
            metrics_df.style.format({
                'Annual Return': '{:.2%}',
                'Annual Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Sortino Ratio': '{:.2f}',
                'Calmar Ratio': '{:.2f}',
                'Omega Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                'VaR (95%)': '{:.2%}',
                'CVaR (95%)': '{:.2%}',
                'VaR (99%)': '{:.2%}',
                'CVaR (99%)': '{:.2%}',
                'Skewness': '{:.2f}',
                'Kurtosis': '{:.2f}',
                'Gain/Loss Ratio': '{:.2f}',
                'Ulcer Index': '{:.3f}',
                'Tail Ratio': '{:.2f}',
                'Positive Months': '{:.1%}'
            }),
            use_container_width=True,
            height=400
        )
    
    # Interactive portfolio construction
    st.subheader("🎛️ Interactive Portfolio Construction")
    
    # Initialize equal weights if not set
    if not st.session_state.portfolio_weights:
        n_assets = len(returns_df.columns)
        equal_weight = 1 / n_assets
        st.session_state.portfolio_weights = {asset: equal_weight for asset in returns_df.columns}
    
    # Create columns for weight sliders
    col1, col2 = st.columns(2)
    
    current_weights = {}
    total_weight = 0
    
    # Group assets by category for organization
    asset_categories = {}
    for asset in returns_df.columns:
        # Determine category
        category = "Other"
        for cat_name, cat_assets in ASSET_UNIVERSE.items():
            if asset in cat_assets:
                category = cat_name.replace('_', ' ')
                break
        if category not in asset_categories:
            asset_categories[category] = []
        asset_categories[category] = asset
    
    # Display sliders by category
    for category_idx, (category, assets_in_category) in enumerate(asset_categories.items()):
        # Alternate between columns
        current_col = col1 if category_idx % 2 == 0 else col2
        
        with current_col:
            with st.expander(f"{category} ({len(assets_in_category)} assets)", expanded=False):
                for asset in assets_in_category:
                    current_weight = st.session_state.portfolio_weights.get(asset, 0)
                    
                    # Create slider
                    new_weight = st.slider(
                        f"{asset}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_weight),
                        step=0.01,
                        format="%.0%",
                        key=f"weight_{asset}"
                    )
                    
                    current_weights[asset] = new_weight
                    total_weight += new_weight
    
    # Display weight summary
    st.write(f"**Total Weight: {total_weight:.1%}**")
    
    if abs(total_weight - 1.0) > 0.001:
        st.warning(f"Weights don't sum to 100% (current: {total_weight:.1%}). Normalizing...")
        # Normalize weights
        if total_weight > 0:
            current_weights = {k: v/total_weight for k, v in current_weights.items()}
        else:
            current_weights = {k: 1/len(current_weights) for k in current_weights.keys()}
    
    # Update session state
    st.session_state.portfolio_weights = current_weights
    
    # Calculate current portfolio performance
    optimizer = GlobalPortfolioOptimizer(returns_df, prices_df, risk_free_rate)
    current_performance = optimizer.calculate_portfolio_performance(current_weights)
    
    # Display current portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Return", f"{current_performance[0]*100:.2f}%")
    
    with col2:
        st.metric("Current Risk", f"{current_performance[1]*100:.2f}%")
    
    with col3:
        st.metric("Current Sharpe", f"{current_performance[2]:.2f}")
    
    with col4:
        # Calculate diversification
        n_assets = sum(1 for w in current_weights.values() if w > 0.001)
        st.metric("Diversification", f"{n_assets} assets")
    
    # Run optimization if requested
    if run_optimization_btn:
        st.subheader("🎯 Optimization Results")
        
        with st.spinner("Running optimizations..."):
            # Initialize optimizer
            optimizer = GlobalPortfolioOptimizer(returns_df, prices_df, risk_free_rate)
            
            # Store results
            optimization_results = {}
            optimal_points = {}
            
            # Run selected strategies
            for strategy_name, selected in strategies.items():
                if not selected:
                    continue
                
                try:
                    if strategy_name == "Equal Weight":
                        weights, performance, label = optimizer.optimize_equal_weight()
                    elif strategy_name == "Max Sharpe Ratio":
                        weights, performance, label = optimizer.optimize_max_sharpe()
                    elif strategy_name == "Min Volatility":
                        weights, performance, label = optimizer.optimize_min_volatility()
                    elif strategy_name == "Max Diversification":
                        weights, performance, label = optimizer.optimize_max_diversification()
                    elif strategy_name == "Risk Parity":
                        weights, performance, label = optimizer.optimize_risk_parity()
                    elif strategy_name == "Max Return":
                        weights, performance, label = optimizer.optimize_max_return()
                    
                    optimization_results[strategy_name] = {
                        'weights': weights,
                        'performance': performance,
                        'label': label
                    }
                    
                    optimal_points[strategy_name] = {
                        'return': performance[0],
                        'volatility': performance[1],
                        'sharpe': performance[2],
                        'returns_df': returns_df,
                        'prices_df': prices_df
                    }
                    
                except Exception as e:
                    st.warning(f"Strategy {strategy_name} failed: {str(e)}")
            
            # Run constrained optimization if selected
            if use_constraints:
                try:
                    weights, performance, label = optimizer.optimize_custom_constraints(max_weight, min_weight)
                    strategy_name = f"Constrained (max {max_weight:.0%})"
                    optimization_results[strategy_name] = {
                        'weights': weights,
                        'performance': performance,
                        'label': label
                    }
                    
                    optimal_points[strategy_name] = {
                        'return': performance[0],
                        'volatility': performance[1],
                        'sharpe': performance[2],
                        'returns_df': returns_df,
                        'prices_df': prices_df
                    }
                except Exception as e:
                    st.warning(f"Constrained optimization failed: {e}")
            
            # Store in session state
            st.session_state.optimization_results = optimization_results
            st.session_state.optimal_points = optimal_points
            
            st.success(f"✅ Completed {len(optimization_results)} optimizations")
    
    # Display optimization results if available
    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        st.subheader("📊 Optimization Comparison")
        
        # Create comparison table
        comparison_data = []
        strategies_weights = {}
        
        for strategy_name, result in st.session_state.optimization_results.items():
            weights = result['weights']
            performance = result['performance']
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Return': performance[0],
                'Risk': performance[1],
                'Sharpe': performance[2],
                'Assets': sum(1 for w in weights.values() if w > 0.001),
                'Max Weight': max(weights.values()) if weights else 0
            })
            
            strategies_weights[strategy_name] = weights
        
        # Add current portfolio
        comparison_data.append({
            'Strategy': 'Current Portfolio',
            'Return': current_performance[0],
            'Risk': current_performance[1],
            'Sharpe': current_performance[2],
            'Assets': sum(1 for w in current_weights.values() if w > 0.001),
            'Max Weight': max(current_weights.values()) if current_weights else 0
        })
        
        strategies_weights['Current Portfolio'] = current_weights
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(
            comparison_df.style.format({
                'Return': '{:.2%}',
                'Risk': '{:.2%}',
                'Sharpe': '{:.2f}',
                'Max Weight': '{:.1%}'
            }).highlight_max(subset=['Sharpe', 'Return'], color='lightgreen')
            .highlight_min(subset=['Risk'], color='lightcoral'),
            use_container_width=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Efficient frontier
            st.subheader("📈 Efficient Frontier")
            
            # Generate frontier and Monte Carlo
            frontier_df = optimizer.generate_efficient_frontier(points=100)
            monte_carlo_df = optimizer.monte_carlo_simulation(n_portfolios=5000)
            
            # Create chart
            fig = create_interactive_frontier_chart(
                frontier_df, 
                monte_carlo_df, 
                st.session_state.get('optimal_points', {}),
                current_weights
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Allocation sunburst
            st.subheader("📊 Portfolio Allocation")
            
            # Use best Sharpe portfolio for allocation display
            best_sharpe_strategy = None
            best_sharpe_value = -np.inf
            
            for strategy_name, result in st.session_state.optimization_results.items():
                if result['performance'][2] > best_sharpe_value:
                    best_sharpe_value = result['performance'][2]
                    best_sharpe_strategy = strategy_name
            
            if best_sharpe_strategy:
                best_weights = st.session_state.optimization_results[best_sharpe_strategy]['weights']
                
                # Create asset categories mapping
                asset_to_category = {}
                for category, assets in ASSET_UNIVERSE.items():
                    for asset in assets:
                        if asset in returns_df.columns:
                            asset_to_category[asset] = category.replace('_', ' ')
                
                # Add any missing assets
                for asset in returns_df.columns:
                    if asset not in asset_to_category:
                        asset_to_category[asset] = "Other"
                
                fig = create_allocation_sunburst(best_weights, asset_to_category)
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison chart
        st.subheader("📈 Strategy Performance Comparison")
        fig = create_performance_comparison_chart(returns_df, strategies_weights, prices_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis by asset class
        st.subheader("⚠️ Risk Analysis by Asset Class")
        
        # Calculate risk metrics by category
        risk_metrics_by_category = {}
        
        for strategy_name, weights in strategies_weights.items():
            if strategy_name == 'Current Portfolio':
                continue
                
            # Group assets by category
            category_weights = {}
            for asset, weight in weights.items():
                if weight < 0.001:
                    continue
                
                # Find category
                category = "Other"
                for cat_name, cat_assets in ASSET_UNIVERSE.items():
                    if asset in cat_assets:
                        category = cat_name.replace('_', ' ')
                        break
                
                if category not in category_weights:
                    category_weights[category] = {}
                category_weights[category][asset] = weight
            
            # Calculate metrics for each category
            for category, cat_weights in category_weights.items():
                if not cat_weights:
                    continue
                
                # Calculate weighted returns for category
                cat_assets = list(cat_weights.keys())
                cat_returns = returns_df[cat_assets]
                
                # Normalize weights within category
                cat_total = sum(cat_weights.values())
                normalized_weights = {k: v/cat_total for k, v in cat_weights.items()}
                
                # Calculate category returns
                weight_array = np.array([normalized_weights[asset] for asset in cat_assets])
                category_returns = cat_returns.dot(weight_array)
                
                # Calculate risk metrics
                var_95 = np.percentile(category_returns, 5)
                cvar_95 = category_returns[category_returns <= var_95].mean()
                
                # Maximum drawdown
                cumulative = (1 + category_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                if category not in risk_metrics_by_category:
                    risk_metrics_by_category[category] = {}
                
                risk_metrics_by_category[category][strategy_name] = {
                    'VaR (95%)': var_95,
                    'CVaR (95%)': cvar_95,
                    'Max Drawdown': max_drawdown
                }
        
        # Create risk metrics chart
        if risk_metrics_by_category:
            # For simplicity, show metrics for best Sharpe strategy
            if best_sharpe_strategy in [list(cat_metrics.keys())[0] for cat_metrics in risk_metrics_by_category.values()]:
                # Extract metrics for best strategy
                best_strategy_metrics = {}
                for category, cat_metrics in risk_metrics_by_category.items():
                    if best_sharpe_strategy in cat_metrics:
                        best_strategy_metrics[category] = cat_metrics[best_sharpe_strategy]
                
                fig = create_risk_metrics_chart(best_strategy_metrics)
                st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        st.subheader("📉 Portfolio Drawdown Analysis")
        
        # Show drawdown for best Sharpe and current portfolio
        col1, col2 = st.columns(2)
        
        with col1:
            if best_sharpe_strategy:
                st.write(f"**{best_sharpe_strategy} Portfolio**")
                fig = create_drawdown_chart(returns_df, strategies_weights[best_sharpe_strategy])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write(f"**Current Portfolio**")
            fig = create_drawdown_chart(returns_df, current_weights)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export weights
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'risk_free_rate': risk_free_rate,
                'period': period,
                'strategies': {}
            }
            
            for strategy_name, result in st.session_state.optimization_results.items():
                export_data['strategies'][strategy_name] = {
                    'weights': result['weights'],
                    'performance': {
                        'return': result['performance'][0],
                        'risk': result['performance'][1],
                        'sharpe': result['performance'][2]
                    }
                }
            
            st.download_button(
                label="📊 Download All Results (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name="portfolio_optimization_results.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export current portfolio
            current_data = {
                'weights': current_weights,
                'performance': {
                    'return': current_performance[0],
                    'risk': current_performance[1],
                    'sharpe': current_performance[2]
                },
                'assets': list(current_weights.keys())
            }
            
            st.download_button(
                label="📥 Download Current Portfolio",
                data=json.dumps(current_data, indent=2),
                file_name="current_portfolio.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Export metrics
            metrics_data = metrics_df.to_dict()
            st.download_button(
                label="📈 Download Asset Metrics",
                data=json.dumps(metrics_data, indent=2),
                file_name="asset_metrics.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Global Portfolio Optimizer Pro** • Built with Streamlit, PyPortfolioOpt, and Yahoo Finance
    • Data is for educational purposes only • Past performance does not guarantee future results
    """)

if __name__ == "__main__":
    main()