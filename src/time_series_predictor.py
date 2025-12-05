"""
Time series forecasting for traffic accident risk prediction
Predicts hourly/daily accident counts using ARIMA, Prophet, and LSTM
"""
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Optional imports (install as needed)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. ARIMA models unavailable.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: prophet not installed. Prophet models unavailable.")

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. LSTM models unavailable.")


class TimeSeriesPredictor:
    """
    Predict accident counts using time series models
    """

    def __init__(self, model_type='arima'):
        """
        Initialize predictor

        Args:
            model_type: 'arima', 'sarima', 'prophet', or 'lstm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None

    def prepare_hourly_data(self, collision_df):
        """
        Aggregate accidents by hour

        Args:
            collision_df: DataFrame with 'date' and 'time' columns

        Returns:
            pd.Series: Hourly accident counts indexed by datetime
        """
        # Combine date and time
        collision_df['datetime'] = pd.to_datetime(
            collision_df['date'] + ' ' + collision_df['time'],
            errors='coerce'
        )

        # Remove invalid datetimes
        collision_df = collision_df.dropna(subset=['datetime'])

        # Aggregate by hour
        hourly_counts = collision_df.groupby(
            collision_df['datetime'].dt.floor('H')
        ).size()

        # Fill missing hours with 0
        full_range = pd.date_range(
            start=hourly_counts.index.min(),
            end=hourly_counts.index.max(),
            freq='H'
        )
        hourly_counts = hourly_counts.reindex(full_range, fill_value=0)

        return hourly_counts

    def prepare_daily_data(self, collision_df):
        """
        Aggregate accidents by day

        Args:
            collision_df: DataFrame with 'date' column

        Returns:
            pd.Series: Daily accident counts indexed by date
        """
        collision_df['date'] = pd.to_datetime(collision_df['date'], errors='coerce')
        collision_df = collision_df.dropna(subset=['date'])

        daily_counts = collision_df.groupby(
            collision_df['date'].dt.date
        ).size()

        # Convert index to datetime
        daily_counts.index = pd.to_datetime(daily_counts.index)

        # Fill missing days
        full_range = pd.date_range(
            start=daily_counts.index.min(),
            end=daily_counts.index.max(),
            freq='D'
        )
        daily_counts = daily_counts.reindex(full_range, fill_value=0)

        return daily_counts

    def train_arima(self, time_series, order=(5, 1, 2)):
        """
        Train ARIMA model

        Args:
            time_series: pd.Series with datetime index
            order: (p, d, q) parameters

        Returns:
            dict: Training metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMA")

        print(f"\nTraining ARIMA{order} model...")
        print(f"Time series length: {len(time_series)}")
        print(f"Date range: {time_series.index.min()} to {time_series.index.max()}")

        # Train model
        self.model = ARIMA(time_series, order=order)
        fitted = self.model.fit()
        self.fitted_model = fitted

        # In-sample predictions
        predictions = fitted.predict(start=0, end=len(time_series)-1)

        # Calculate metrics
        mae = np.mean(np.abs(time_series - predictions))
        rmse = np.sqrt(np.mean((time_series - predictions) ** 2))
        mape = np.mean(np.abs((time_series - predictions) / (time_series + 1))) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'aic': fitted.aic,
            'bic': fitted.bic
        }

        print(f"\nARIMA Model Performance:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  AIC:  {fitted.aic:.2f}")
        print(f"  BIC:  {fitted.bic:.2f}")

        return metrics

    def train_sarima(self, time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
        """
        Train SARIMA model (seasonal ARIMA)

        Args:
            time_series: pd.Series with datetime index
            order: (p, d, q) parameters
            seasonal_order: (P, D, Q, s) parameters

        Returns:
            dict: Training metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for SARIMA")

        print(f"\nTraining SARIMA{order}x{seasonal_order} model...")

        # Train model
        self.model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
        fitted = self.model.fit(disp=False)
        self.fitted_model = fitted

        # In-sample predictions
        predictions = fitted.predict(start=0, end=len(time_series)-1)

        # Calculate metrics
        mae = np.mean(np.abs(time_series - predictions))
        rmse = np.sqrt(np.mean((time_series - predictions) ** 2))
        mape = np.mean(np.abs((time_series - predictions) / (time_series + 1))) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'aic': fitted.aic,
            'bic': fitted.bic
        }

        print(f"\nSARIMA Model Performance:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  AIC:  {fitted.aic:.2f}")

        return metrics

    def train_prophet(self, time_series):
        """
        Train Facebook Prophet model

        Args:
            time_series: pd.Series with datetime index

        Returns:
            dict: Training metrics
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet required. Install: pip install prophet")

        print(f"\nTraining Prophet model...")

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': time_series.index,
            'y': time_series.values
        })

        # Train model with weekly and daily seasonality
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.model.fit(df)

        # In-sample predictions
        forecast = self.model.predict(df)
        predictions = forecast['yhat'].values

        # Calculate metrics
        mae = np.mean(np.abs(time_series.values - predictions))
        rmse = np.sqrt(np.mean((time_series.values - predictions) ** 2))
        mape = np.mean(np.abs((time_series.values - predictions) / (time_series.values + 1))) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

        print(f"\nProphet Model Performance:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def train_lstm(self, time_series, sequence_length=24, epochs=50):
        """
        Train LSTM model (requires PyTorch)

        Args:
            time_series: pd.Series with datetime index
            sequence_length: Number of past timesteps to use
            epochs: Training epochs

        Returns:
            dict: Training metrics
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM")

        print(f"\nTraining LSTM model...")
        print(f"Sequence length: {sequence_length}")
        print(f"Epochs: {epochs}")

        # Normalize data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(time_series.values.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length])

        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))

        # Define LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=50, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out

        # Train model
        self.model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).numpy()

        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions)
        actuals = self.scaler.inverse_transform(y.numpy())

        # Calculate metrics
        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1))) * 100

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'sequence_length': sequence_length
        }

        print(f"\nLSTM Model Performance:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def forecast(self, steps=24):
        """
        Forecast future accident counts

        Args:
            steps: Number of time steps to forecast

        Returns:
            np.array: Forecasted values
        """
        if self.model_type == 'arima' or self.model_type == 'sarima':
            if not hasattr(self, 'fitted_model'):
                raise ValueError("Model not trained yet")
            forecast = self.fitted_model.forecast(steps=steps)
            return forecast

        elif self.model_type == 'prophet':
            if self.model is None:
                raise ValueError("Model not trained yet")
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps, freq='H')
            forecast = self.model.predict(future)
            return forecast['yhat'].values[-steps:]

        elif self.model_type == 'lstm':
            # TODO: Implement LSTM forecasting
            raise NotImplementedError("LSTM forecasting not yet implemented")

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def save_model(self, filepath='models/timeseries_model.pkl'):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if self.model_type in ['arima', 'sarima']:
            joblib.dump({
                'model_type': self.model_type,
                'fitted_model': self.fitted_model
            }, filepath)
        elif self.model_type == 'prophet':
            joblib.dump({
                'model_type': self.model_type,
                'model': self.model
            }, filepath)
        elif self.model_type == 'lstm':
            torch.save({
                'model_type': self.model_type,
                'model_state': self.model.state_dict(),
                'scaler': self.scaler
            }, filepath)

        print(f"\nModel saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load trained model"""
        if filepath.endswith('.pkl'):
            data = joblib.load(filepath)
        else:
            data = torch.load(filepath)

        predictor = TimeSeriesPredictor(model_type=data['model_type'])

        if data['model_type'] in ['arima', 'sarima']:
            predictor.fitted_model = data['fitted_model']
        elif data['model_type'] == 'prophet':
            predictor.model = data['model']
        elif data['model_type'] == 'lstm':
            predictor.model.load_state_dict(data['model_state'])
            predictor.scaler = data['scaler']

        return predictor


def compare_timeseries_models(time_series, frequency='daily'):
    """
    Compare all time series models

    Args:
        time_series: pd.Series with datetime index
        frequency: 'daily' or 'hourly'

    Returns:
        dict: Results for all models
    """
    results = {}

    # ARIMA
    if STATSMODELS_AVAILABLE:
        print("\n" + "="*60)
        print("ARIMA Model")
        print("="*60)
        predictor = TimeSeriesPredictor('arima')
        try:
            metrics = predictor.train_arima(time_series, order=(5, 1, 2))
            results['ARIMA'] = metrics
        except Exception as e:
            print(f"ARIMA training failed: {e}")

    # SARIMA (only for hourly data)
    if STATSMODELS_AVAILABLE and frequency == 'hourly':
        print("\n" + "="*60)
        print("SARIMA Model (with 24-hour seasonality)")
        print("="*60)
        predictor = TimeSeriesPredictor('sarima')
        try:
            metrics = predictor.train_sarima(time_series,
                                            order=(1, 1, 1),
                                            seasonal_order=(1, 1, 1, 24))
            results['SARIMA'] = metrics
        except Exception as e:
            print(f"SARIMA training failed: {e}")

    # Prophet
    if PROPHET_AVAILABLE:
        print("\n" + "="*60)
        print("Prophet Model")
        print("="*60)
        predictor = TimeSeriesPredictor('prophet')
        try:
            metrics = predictor.train_prophet(time_series)
            results['Prophet'] = metrics
        except Exception as e:
            print(f"Prophet training failed: {e}")

    # Print summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY - Model Comparison")
        print("="*60)
        print(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:<15} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} {metrics['mape']:<10.2f}%")

    return results
