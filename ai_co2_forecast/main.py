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

def load_data(file_path):
    """
    Load CO2 emissions data from CSV file.
    If not found, fetch from World Bank API.
    """
    # Ensure data directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Fetching data from World Bank...")
        try:
            # URL for CO2 emissions data download (ZIP file)
            url = "https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.KT?downloadformat=csv"
            zip_path = file_path.replace('.csv', '.zip')

            # Download the ZIP file
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad status codes

            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print("ZIP file downloaded successfully.")

            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(file_path))
            print("ZIP file extracted.")

            # Find and rename the CSV file
            data_dir = os.path.dirname(file_path)
            for file in os.listdir(data_dir):
                if file.endswith('.csv') and 'EN.ATM.CO2E.KT' in file:
                    os.rename(os.path.join(data_dir, file), file_path)
                    print(f"CSV file renamed to {os.path.basename(file_path)}")
                    break

            # Load the data
            df = pd.read_csv(file_path, skiprows=4, encoding='utf-8')  # Skip metadata rows
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the CO2 emissions data.
    Reshape from wide to long format and select a country for forecasting.
    """
    print("Preprocessing data...")

    # Select a country for forecasting (e.g., United States)
    # You can change this to any country or 'World' if available
    country = 'United States'

    # Filter for the selected country
    df_country = df[df['Country Name'] == country]

    if df_country.empty:
        print(f"Country '{country}' not found. Using World data if available...")
        df_country = df[df['Country Name'] == 'World']
        if df_country.empty:
            print("World data not found. Using the first available country.")
            df_country = df.iloc[0:1]  # Take first row

    # Melt the dataframe to long format
    id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    value_vars = [col for col in df.columns if col.isdigit()]  # Year columns
    df_long = df_country.melt(id_vars=id_vars, value_vars=value_vars,
                              var_name='Year', value_name='CO2_Emissions')

    # Convert Year to int and drop rows with NaN emissions
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long = df_long.dropna(subset=['CO2_Emissions'])

    # Sort by year
    df_long = df_long.sort_values('Year')

    print(f"Preprocessed data for {df_long['Country Name'].iloc[0]}. Shape: {df_long.shape}")
    print(f"Years range: {df_long['Year'].min()} - {df_long['Year'].max()}")

    return df_long[['Year', 'CO2_Emissions']]

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
    Evaluate both Linear Regression and Random Forest models.
    """
    # Evaluate Linear Regression
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print("Linear Regression Results:")
    print(f"  Mean Squared Error: {mse_lr:.2f}")
    print(f"  R-squared Score: {r2_lr:.2f}")

    # Evaluate Random Forest
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print("\nRandom Forest Results:")
    print(f"  Mean Squared Error: {mse_rf:.2f}")
    print(f"  R-squared Score: {r2_rf:.2f}")

    return mse_lr, r2_lr, mse_rf, r2_rf, y_pred_lr, y_pred_rf

def forecast_future(lr_model, rf_model, df_processed):
    """
    Forecast CO2 emissions for future years using both models.
    """
    future_years = np.array([[2025], [2030], [2035], [2040]])
    future_predictions_lr = lr_model.predict(future_years)
    future_predictions_rf = rf_model.predict(future_years)

    print("\nFuture CO₂ Emissions Predictions:")
    for year, pred_lr, pred_rf in zip(future_years.flatten(), future_predictions_lr, future_predictions_rf):
        print(f"Year {year}: Linear Regression: {pred_lr:.2f}, Random Forest: {pred_rf:.2f}")

    return future_years, future_predictions_lr, future_predictions_rf

def plot_results(y_test, y_pred_lr, y_pred_rf, df_processed, future_years, future_predictions_lr, future_predictions_rf):
    """
    Plot the actual vs predicted values and future forecasts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot for actual vs predicted
    ax1.scatter(y_test, y_pred_lr, alpha=0.7, color='blue', label='Linear Regression')
    ax1.scatter(y_test, y_pred_rf, alpha=0.7, color='green', label='Random Forest')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual CO2 Emissions')
    ax1.set_ylabel('Predicted CO2 Emissions')
    ax1.set_title('Actual vs Predicted CO2 Emissions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Line plot for historical and future data
    ax2.plot(df_processed['Year'], df_processed['CO2_Emissions'], 'bo-', label='Historical Data', markersize=4)
    ax2.plot(future_years.flatten(), future_predictions_lr, 'r--o', label='Linear Regression Forecast', markersize=6)
    ax2.plot(future_years.flatten(), future_predictions_rf, 'g--s', label='Random Forest Forecast', markersize=6)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('CO₂ Emissions (metric tons)')
    ax2.set_title('CO₂ Emissions: Historical Data and Future Forecasts')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('co2_emissions_forecast.png')
    plt.show()

def main():
    """
    Main function to run the CO2 emissions forecasting pipeline.
    """
    # File path to the dataset
    data_file = 'data/co2_emissions.csv'

    # Load data
    df = load_data(data_file)
    if df is None:
        return

    # Preprocess data
    df_processed = preprocess_data(df)

    # For demonstration, let's assume we have features and target
    # In a real scenario, you would select appropriate features
    # Here we're using a placeholder
    if len(df_processed.columns) < 2:
        print("Warning: Not enough columns for modeling. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        np.random.seed(42)
        X = np.random.rand(100, 1)
        y = 2 * X.flatten() + np.random.randn(100) * 0.1
        df_processed = pd.DataFrame({'feature': X.flatten(), 'target': y})

    # Split data into features and target
    X = df_processed.iloc[:, :-1].values
    y = df_processed.iloc[:, -1].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    lr_model, rf_model = train_models(X_train, y_train)

    # Evaluate models
    mse_lr, r2_lr, mse_rf, r2_rf, y_pred_lr, y_pred_rf = evaluate_models(lr_model, rf_model, X_test, y_test)

    # Forecast future emissions
    future_years, future_predictions_lr, future_predictions_rf = forecast_future(lr_model, rf_model, df_processed)

    # Plot results
    plot_results(y_test, y_pred_lr, y_pred_rf, df_processed, future_years, future_predictions_lr, future_predictions_rf)

    print("CO2 emissions forecasting completed!")

if __name__ == "__main__":
    main()
