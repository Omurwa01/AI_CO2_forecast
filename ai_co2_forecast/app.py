gitimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import requests
import zipfile
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="CO₂ Emissions Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Load CO2 emissions data from OWID dataset.
    """
    with st.spinner("Loading CO2 emissions data..."):
        try:
            # Use OWID CO2 dataset (cleaner format)
            url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
            df = pd.read_csv(url)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

@st.cache_data
def preprocess_data(df, country):
    """
    Preprocess the CO2 emissions data for a selected country.
    """
    # Filter for the selected country
    df_country = df[df['country'] == country]

    if df_country.empty:
        return None

    # Select relevant columns and drop NaN values
    df_processed = df_country[['year', 'co2']].dropna()

    # Rename columns for consistency
    df_processed = df_processed.rename(columns={'year': 'Year', 'co2': 'CO2_Emissions'})

    # Sort by year
    df_processed = df_processed.sort_values('Year')

    return df_processed

@st.cache_data
def train_models(X_train, y_train):
    """
    Train Linear Regression and Random Forest models.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return lr_model, rf_model

def evaluate_models(lr_model, rf_model, X_test, y_test):
    """
    Evaluate both models and return metrics.
    """
    # Evaluate Linear Regression
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    return {
        'lr': {'mse': mse_lr, 'r2': r2_lr, 'predictions': y_pred_lr},
        'rf': {'mse': mse_rf, 'r2': r2_rf, 'predictions': y_pred_rf}
    }

def forecast_emissions(lr_model, rf_model, start_year, end_year):
    """
    Forecast CO2 emissions for the specified year range.
    """
    future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)

    lr_predictions = lr_model.predict(future_years)
    rf_predictions = rf_model.predict(future_years)

    forecast_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Linear_Regression': lr_predictions,
        'Random_Forest': rf_predictions
    })

    return forecast_df

def create_forecast_plot(historical_df, forecast_df):
    """
    Create an interactive plot showing historical and forecasted data.
    """
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_df['Year'],
            y=historical_df['CO2_Emissions'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ),
        secondary_y=False
    )

    # Linear Regression forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Linear_Regression'],
            mode='lines+markers',
            name='Linear Regression Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ),
        secondary_y=False
    )

    # Random Forest forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Random_Forest'],
            mode='lines+markers',
            name='Random Forest Forecast',
            line=dict(color='#2ca02c', width=3, dash='dot'),
            marker=dict(size=8, symbol='square')
        ),
        secondary_y=False
    )

    # Update layout
    fig.update_layout(
        title={
            'text': 'CO₂ Emissions: Historical Data and Forecasts',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (metric tons)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

def create_comparison_plot(historical_df1, forecast_df1, historical_df2, forecast_df2, country1, country2):
    """
    Create an interactive plot comparing two countries' historical and forecasted data.
    """
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Country 1 - Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_df1['Year'],
            y=historical_df1['CO2_Emissions'],
            mode='lines+markers',
            name=f'{country1} - Historical',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ),
        secondary_y=False
    )

    # Country 1 - Linear Regression forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df1['Year'],
            y=forecast_df1['Linear_Regression'],
            mode='lines+markers',
            name=f'{country1} - LR Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ),
        secondary_y=False
    )

    # Country 1 - Random Forest forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df1['Year'],
            y=forecast_df1['Random_Forest'],
            mode='lines+markers',
            name=f'{country1} - RF Forecast',
            line=dict(color='#2ca02c', width=3, dash='dot'),
            marker=dict(size=8, symbol='square')
        ),
        secondary_y=False
    )

    # Country 2 - Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_df2['Year'],
            y=historical_df2['CO2_Emissions'],
            mode='lines+markers',
            name=f'{country2} - Historical',
            line=dict(color='#d62728', width=3),
            marker=dict(size=6)
        ),
        secondary_y=False
    )

    # Country 2 - Linear Regression forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df2['Year'],
            y=forecast_df2['Linear_Regression'],
            mode='lines+markers',
            name=f'{country2} - LR Forecast',
            line=dict(color='#9467bd', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ),
        secondary_y=False
    )

    # Country 2 - Random Forest forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df2['Year'],
            y=forecast_df2['Random_Forest'],
            mode='lines+markers',
            name=f'{country2} - RF Forecast',
            line=dict(color='#8c564b', width=3, dash='dot'),
            marker=dict(size=8, symbol='square')
        ),
        secondary_y=False
    )

    # Update layout
    fig.update_layout(
        title={
            'text': f'CO₂ Emissions Comparison: {country1} vs {country2}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis_title='Year',
        yaxis_title='CO₂ Emissions (metric tons)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

def main():
    """
    Main Streamlit application.
    """
    # Main header
    st.markdown('<h1 class="main-header">🌍 CO₂ Emissions Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Predict future CO₂ emissions using machine learning models for climate action planning.")

    # Sidebar
    st.sidebar.markdown('<h2 class="sidebar-header">🔧 Controls</h2>', unsafe_allow_html=True)

    # Load data
    df = load_data()

    if df is None:
        st.error("Failed to load data. Please check your internet connection and try again.")
        return

    # Country selection
    countries = sorted(df['country'].dropna().unique())

    # Single country analysis
    st.sidebar.markdown("### Single Country Analysis")
    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries,
        index=countries.index('United States') if 'United States' in countries else 0,
        key='single_country'
    )

    # Year range selection
    current_year = pd.Timestamp.now().year
    start_year = st.sidebar.slider("Forecast Start Year", current_year + 1, current_year + 20, current_year + 5)
    end_year = st.sidebar.slider("Forecast End Year", start_year, current_year + 30, current_year + 15)

    # Country comparison section
    st.sidebar.markdown("### Country Comparison")
    country1 = st.sidebar.selectbox(
        "Select Country A",
        countries,
        index=countries.index('United States') if 'United States' in countries else 0,
        key='country1'
    )
    country2 = st.sidebar.selectbox(
        "Select Country B",
        countries,
        index=countries.index('China') if 'China' in countries else 1,
        key='country2'
    )

    # Tabs for different views
    tab1, tab2 = st.tabs(["📊 Single Country Analysis", "🔄 Country Comparison"])

    with tab1:
        # Single country analysis
        st.markdown(f"## Analysis for {selected_country}")

        # Preprocess data for selected country
        df_processed = preprocess_data(df, selected_country)

        if df_processed is None or len(df_processed) < 5:
            st.error(f"Insufficient data for {selected_country}. Please select another country.")
        else:
            # Split data
            X = df_processed[['Year']].values
            y = df_processed['CO2_Emissions'].values

            if len(X) < 10:
                st.warning("Limited historical data. Results may not be reliable.")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train models
            with st.spinner("Training models..."):
                lr_model, rf_model = train_models(X_train, y_train)

            # Evaluate models
            metrics = evaluate_models(lr_model, rf_model, X_test, y_test)

            # Forecast
            forecast_df = forecast_emissions(lr_model, rf_model, start_year, end_year)

            # Main content
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Linear Regression R²", f"{metrics['lr']['r2']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Random Forest R²", f"{metrics['rf']['r2']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("LR MSE", f"{metrics['lr']['mse']:.2e}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RF MSE", f"{metrics['rf']['mse']:.2e}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Forecast results
            st.subheader("📈 Forecast Results")
            st.dataframe(forecast_df.style.format({
                'Linear_Regression': '{:.2f}',
                'Random_Forest': '{:.2f}'
            }))

            # Visualization
            st.subheader("📊 Historical vs Forecasted Emissions")
            fig = create_forecast_plot(df_processed, forecast_df)
            st.plotly_chart(fig, width='stretch')

            # Model comparison
            st.subheader("⚖️ Model Comparison")
            comparison_data = {
                'Model': ['Linear Regression', 'Random Forest'],
                'R² Score': [metrics['lr']['r2'], metrics['rf']['r2']],
                'MSE': [metrics['lr']['mse'], metrics['rf']['mse']]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df.style.format({
                'R² Score': '{:.3f}',
                'MSE': '{:.2e}'
            }))

    with tab2:
        # Country comparison
        st.markdown(f"## Comparison: {country1} vs {country2}")

        # Preprocess data for both countries
        df_processed1 = preprocess_data(df, country1)
        df_processed2 = preprocess_data(df, country2)

        if df_processed1 is None or len(df_processed1) < 5:
            st.error(f"Insufficient data for {country1}.")
        elif df_processed2 is None or len(df_processed2) < 5:
            st.error(f"Insufficient data for {country2}.")
        else:
            # Process country 1
            X1 = df_processed1[['Year']].values
            y1 = df_processed1['CO2_Emissions'].values

            if len(X1) < 10:
                X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.5, random_state=42)
            else:
                X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

            with st.spinner(f"Training models for {country1}..."):
                lr_model1, rf_model1 = train_models(X_train1, y_train1)

            metrics1 = evaluate_models(lr_model1, rf_model1, X_test1, y_test1)
            forecast_df1 = forecast_emissions(lr_model1, rf_model1, 2025, 2040)

            # Process country 2
            X2 = df_processed2[['Year']].values
            y2 = df_processed2['CO2_Emissions'].values

            if len(X2) < 10:
                X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.5, random_state=42)
            else:
                X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

            with st.spinner(f"Training models for {country2}..."):
                lr_model2, rf_model2 = train_models(X_train2, y_train2)

            metrics2 = evaluate_models(lr_model2, rf_model2, X_test2, y_test2)
            forecast_df2 = forecast_emissions(lr_model2, rf_model2, 2025, 2040)

            # Display metrics for both countries
            st.subheader("📊 Model Performance Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {country1}")
                st.metric("Linear Regression R²", f"{metrics1['lr']['r2']:.3f}")
                st.metric("Random Forest R²", f"{metrics1['rf']['r2']:.3f}")
                st.metric("LR MSE", f"{metrics1['lr']['mse']:.2e}")
                st.metric("RF MSE", f"{metrics1['rf']['mse']:.2e}")

            with col2:
                st.markdown(f"### {country2}")
                st.metric("Linear Regression R²", f"{metrics2['lr']['r2']:.3f}")
                st.metric("Random Forest R²", f"{metrics2['rf']['r2']:.3f}")
                st.metric("LR MSE", f"{metrics2['lr']['mse']:.2e}")
                st.metric("RF MSE", f"{metrics2['rf']['mse']:.2e}")

            # Comparison visualization
            st.subheader("📈 Country Comparison (2025-2040 Forecasts)")
            fig_comparison = create_comparison_plot(df_processed1, forecast_df1, df_processed2, forecast_df2, country1, country2)
            st.plotly_chart(fig_comparison, width='stretch')

            # Forecast tables
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"#### {country1} Forecast (2025-2040)")
                st.dataframe(forecast_df1.style.format({
                    'Linear_Regression': '{:.2f}',
                    'Random_Forest': '{:.2f}'
                }))

            with col2:
                st.markdown(f"#### {country2} Forecast (2025-2040)")
                st.dataframe(forecast_df2.style.format({
                    'Linear_Regression': '{:.2f}',
                    'Random_Forest': '{:.2f}'
                }))

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ for Climate Action | Data source: Our World in Data (OWID)")

if __name__ == "__main__":
    main()
