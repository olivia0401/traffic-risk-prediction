"""
Time series forecasting for traffic accident risk prediction
Predicts hourly/daily accident counts using ARIMA, Prophet, and LSTM
"""
import pandas as pd
import numpy as np
import warnings
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
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
    PYTORCH_AVAILABLE = True

    class LSTMForecaster(nn.Module):
        """Two-layer LSTM regressor for univariate accident-count forecasting."""

        def __init__(self, input_size=1, hidden_size=50, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

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
        # Combine date and time (UK DfT dates are DD/MM/YYYY → dayfirst)
        collision_df['datetime'] = pd.to_datetime(
            collision_df['date'] + ' ' + collision_df['time'],
            dayfirst=True, errors='coerce'
        )

        # Remove invalid datetimes
        collision_df = collision_df.dropna(subset=['datetime'])

        # Aggregate by hour
        hourly_counts = collision_df.groupby(
            collision_df['datetime'].dt.floor('h')
        ).size()

        # Fill missing hours with 0
        full_range = pd.date_range(
            start=hourly_counts.index.min(),
            end=hourly_counts.index.max(),
            freq='h'
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
        collision_df['date'] = pd.to_datetime(collision_df['date'],
                                              dayfirst=True, errors='coerce')
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

        print("\nARIMA Model Performance:")
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

        print("\nSARIMA Model Performance:")
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

        print("\nTraining Prophet model...")

        # Remember the series frequency so forecast() steps in days for a daily
        # series and in hours for an hourly one.
        self.freq = pd.infer_freq(time_series.index) or 'h'

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

        print("\nProphet Model Performance:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")

        return metrics

    def train_lstm(self, time_series, sequence_length=24, epochs=100,
                   hidden_size=50, num_layers=2, batch_size=32,
                   test_frac=0.2, patience=10, lr=1e-3):
        """
        Train an LSTM sequence forecaster (PyTorch) with an honest chronological
        train / validation / test backtest.

        The series is split in time order — never shuffled — so the model is only
        ever trained on the past and scored on the future:

            [-------- train --------][-- val --][---- test ----]

        The scaler is fit on the training region only (no look-ahead leakage), a
        validation tail drives early stopping, and the reported MAE/RMSE/MAPE are
        computed on the untouched test tail. These are therefore genuine
        one-step-ahead errors, not the in-sample fit the earlier version reported.

        Args:
            time_series: pd.Series with a datetime index
            sequence_length: number of past timesteps fed to the LSTM
            epochs: maximum training epochs (early stopping usually stops sooner)
            hidden_size, num_layers: LSTM capacity
            batch_size: mini-batch size
            test_frac: fraction of the most recent data held out for testing
            patience: early-stopping patience (epochs without val improvement)
            lr: Adam learning rate

        Returns:
            dict: held-out test metrics
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTM")

        print("\nTraining LSTM model...")
        print(f"Sequence length: {sequence_length} | max epochs: {epochs}")

        values = time_series.values.astype("float32").reshape(-1, 1)
        n = len(values)
        if n < sequence_length + 10:
            raise ValueError(
                f"Series too short ({n}) for sequence_length={sequence_length}"
            )

        # --- chronological split; fit scaler on TRAIN only to avoid leakage ---
        n_test = max(1, int(round(n * test_frac)))
        split = n - n_test
        val_split = max(sequence_length + 1, int(round(split * 0.9)))

        self.scaler = MinMaxScaler()
        self.scaler.fit(values[:split])
        scaled = self.scaler.transform(values)

        def make_sequences(arr, start, end):
            """Windows ending at t (exclusive-of-t inputs) predicting value at t."""
            X, y = [], []
            for t in range(max(start, sequence_length), end):
                X.append(arr[t - sequence_length:t])
                y.append(arr[t])
            if not X:
                return (np.empty((0, sequence_length, 1), dtype="float32"),
                        np.empty((0, 1), dtype="float32"))
            return np.array(X, dtype="float32"), np.array(y, dtype="float32")

        X_tr, y_tr = make_sequences(scaled, sequence_length, val_split)
        X_val, y_val = make_sequences(scaled, val_split, split)
        X_te, y_te = make_sequences(scaled, split, n)
        if len(X_te) == 0:
            raise ValueError("Test split produced no samples; raise test_frac or add data")

        torch.manual_seed(42)
        train_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=batch_size, shuffle=True,
        )

        self.model = LSTMForecaster(1, hidden_size, num_layers)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        has_val = len(X_val) > 0
        Xv = torch.from_numpy(X_val) if has_val else None
        yv = torch.from_numpy(y_val) if has_val else None

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(Xv), yv).item()
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.clone()
                                  for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{epochs}]  val_loss={val_loss:.4f}")
                if epochs_no_improve >= patience:
                    print(f"  Early stop at epoch {epoch+1} "
                          f"(best val_loss={best_val:.4f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # keep the final window so forecast() can roll forward from the series end
        self.last_sequence = scaled[-sequence_length:].copy()

        # --- honest held-out evaluation: one-step-ahead on the test tail ---
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(torch.from_numpy(X_te)).numpy()
        predictions = self.scaler.inverse_transform(pred_scaled)
        actuals = self.scaler.inverse_transform(y_te)

        mae = float(np.mean(np.abs(actuals - predictions)))
        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        mape = float(np.mean(np.abs((actuals - predictions) / (actuals + 1))) * 100)

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'sequence_length': sequence_length,
            'n_test': int(len(X_te)),
            'eval': 'held-out chronological test tail',
        }

        print(f"\nLSTM Model Performance (held-out test, n={len(X_te)}):")
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
            future = self.model.make_future_dataframe(
                periods=steps, freq=getattr(self, 'freq', 'h'))
            forecast = self.model.predict(future)
            return forecast['yhat'].values[-steps:]

        elif self.model_type == 'lstm':
            if self.model is None or getattr(self, 'last_sequence', None) is None:
                raise ValueError("Model not trained yet")
            # Recursive multi-step forecast: feed each prediction back in as the
            # newest observation and slide the input window forward one step.
            self.model.eval()
            window = self.last_sequence.copy()          # (seq_len, 1), scaled
            preds_scaled = []
            with torch.no_grad():
                for _ in range(steps):
                    x = torch.from_numpy(
                        window.reshape(1, self.sequence_length, 1).astype("float32")
                    )
                    nxt = self.model(x).numpy().reshape(1, 1)   # scaled next value
                    preds_scaled.append(nxt[0, 0])
                    window = np.vstack([window[1:], nxt])        # roll forward
            return self.scaler.inverse_transform(
                np.array(preds_scaled, dtype="float32").reshape(-1, 1)
            ).ravel()

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
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'hidden_size': getattr(self, 'hidden_size', 50),
                'num_layers': getattr(self, 'num_layers', 2),
                'last_sequence': getattr(self, 'last_sequence', None),
            }, filepath)

        print(f"\nModel saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load trained model"""
        if filepath.endswith('.pkl'):
            data = joblib.load(filepath)
        else:
            data = torch.load(filepath, weights_only=False)

        predictor = TimeSeriesPredictor(model_type=data['model_type'])

        if data['model_type'] in ['arima', 'sarima']:
            predictor.fitted_model = data['fitted_model']
        elif data['model_type'] == 'prophet':
            predictor.model = data['model']
        elif data['model_type'] == 'lstm':
            predictor.sequence_length = data['sequence_length']
            predictor.hidden_size = data.get('hidden_size', 50)
            predictor.num_layers = data.get('num_layers', 2)
            predictor.model = LSTMForecaster(1, predictor.hidden_size,
                                             predictor.num_layers)
            predictor.model.load_state_dict(data['model_state'])
            predictor.model.eval()
            predictor.scaler = data['scaler']
            predictor.last_sequence = data.get('last_sequence')

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

    # LSTM
    if PYTORCH_AVAILABLE:
        print("\n" + "="*60)
        print("LSTM Model (held-out chronological backtest)")
        print("="*60)
        predictor = TimeSeriesPredictor('lstm')
        try:
            seq_len = 24 if frequency == 'hourly' else 7
            metrics = predictor.train_lstm(time_series, sequence_length=seq_len)
            results['LSTM'] = metrics
        except Exception as e:
            print(f"LSTM training failed: {e}")

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
