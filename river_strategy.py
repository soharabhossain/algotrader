#-------------------
# river_strategy.py
#-------------------

import ta, traceback
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf


import math
from datetime import datetime
from river import linear_model, preprocessing, compose, metrics, drift

# Classifiers
from river.linear_model import LogisticRegression
from river.naive_bayes import GaussianNB
from river.tree import HoeffdingTreeClassifier
from river.forest import ARFClassifier
from river.neighbors import KNNClassifier

# Regressors
from river.linear_model import LinearRegression
from river.tree import HoeffdingTreeRegressor
from river.forest import ARFRegressor
from river.neighbors import KNNRegressor

from strategy import Strategy

#-----------------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-----------------------------------------------------------------------------

# This is the implementation of a specific strategy
class IntradayMultiTaskModelStrategy(Strategy):

    def __init__(self, drift_confidence: float = 0.002, 
                 classifier_type: str = "logistic", 
                 regressor_type: str = "linear"):
        super().__init__()         
        
        # ML pipelines
        # Classification Pipeline - Predict price movement direction - Up/Down
        self.classifier_pipeline = lambda: compose.Pipeline(
            preprocessing.StandardScaler(),
            self._get_classifier(classifier_type)
        )
        self.classifier = self.classifier_pipeline()

        # Regression Pipeline - Predict price change (Log return -> target price)
        self.regressor_pipeline = lambda: compose.Pipeline(
            preprocessing.StandardScaler(),
            self._get_regressor(regressor_type)
        )
        self.regressor = self.regressor_pipeline()

        # Drift detectors
        self.classifier_drift = drift.ADWIN(delta=drift_confidence)
        self.regressor_drift = drift.ADWIN(delta=drift_confidence)

        # Metrics
        self.classification_metric = metrics.Accuracy()
        self.regression_metric = metrics.MAE()

        # For tracking
        self.dates = []
        self.actual_prices = []
        self.predicted_prices = []
        self.true_directions = []
        self.predicted_directions = []
        self.confidence_scores = []

    def _get_classifier(self, name):
        return {
            "logistic": LogisticRegression(),
            "hoeffding_tree": HoeffdingTreeClassifier(),
            "ar_forest": ARFClassifier(),
            "knn": KNNClassifier(n_neighbors=10),
            "naive_bayes": GaussianNB()
        }.get(name, LogisticRegression())

    def _get_regressor(self, name):
        return {
            "linear":LinearRegression(),
            "hoeffding_tree": HoeffdingTreeRegressor(),
            "ar_forest": ARFRegressor(),
            "knn": KNNRegressor(n_neighbors=10)
        }.get(name, LinearRegression())

 #--------------------------------------------------------------------------   
    def _compute_features(self, prev_row, curr_row):
        return {
            "price_change": (float(curr_row["Close"]) - float(prev_row["Close"])) / float(prev_row["Close"]),
            "volume_change": (float(curr_row["Volume"]) - float(prev_row["Volume"])) / (float(prev_row["Volume"]) + 1e-8),
            "high_low_range": (float(curr_row["High"]) - float(curr_row["Low"])) / float(curr_row["Open"]),
            "close_open_diff": (float(curr_row["Close"]) - float(curr_row["Open"])) / float(curr_row["Open"]),
        }
 #--------------------------------------------------------------------------   
    def _compute_log_return(self, prev_close, curr_close):
        return math.log(float(curr_close) / float(prev_close))
 #--------------------------------------------------------------------------   
    def _estimate_next_price(self, current_price, log_return):
        return current_price * math.exp(log_return)
 #-------------------------------------------------------------------------- 
 # This is one of the methods of the base class implemented in this subclass  
    def run_on_data(self, prev_row, curr_row, warmup, indicator_skip_count=10, model_skip_count=5) -> dict:
        try:
            x = self._compute_features(prev_row, curr_row)
            y_direction = int(float(curr_row["Close"]) > float(prev_row["Close"]))
            y_log_return = self._compute_log_return(prev_row["Close"], curr_row["Close"])
            current_price = float(prev_row["Close"])
            timestamp = datetime.strptime(curr_row["Date"], "%d-%b-%y")

            # Prediction (for drift and evaluation)
            pred_dir = self.classifier.predict_one(x)
            confidence = round(self.classifier.predict_proba_one(x).get(pred_dir, 0), 4)
            pred_return = self.regressor.predict_one(x)

            # Drift detection
            self.classifier_drift.update(int(pred_dir == y_direction))
            self.regressor_drift.update(abs(pred_return - y_log_return))

            if self.classifier_drift.drift_detected:
                self.classifier = self.classifier_pipeline()
                print(f"[{timestamp}] 📉 Classifier drift detected. Resetting model.")
            if self.regressor_drift.drift_detected:
                self.regressor = self.regressor_pipeline()
                print(f"[{timestamp}] 📉 Regressor drift detected. Resetting model.")

            # Train
            self.classifier.learn_one(x, y_direction)
            self.regressor.learn_one(x, y_log_return)

            # Update metrics
            self.classification_metric.update(y_direction, pred_dir)
            self.regression_metric.update(y_log_return, pred_return)

            # Save for plotting
            timestamp = datetime.strptime(curr_row["Date"], "%d-%b-%y")
            self.dates.append(timestamp)
            actual_price = float(curr_row["Close"])
            predicted_price = self._estimate_next_price(current_price, pred_return)
            self.actual_prices.append(actual_price)
            self.predicted_prices.append(predicted_price)
            self.true_directions.append("Up" if y_direction else "Down")
            self.predicted_directions.append("Up" if pred_dir else "Down")
            self.confidence_scores.append(confidence)

            # return {
            #     "timestamp": timestamp,
            #     "actual_price": actual_price,
            #     "predicted_price": predicted_price,
            #     "log_return": round(pred_return, 4),
            #     "direction": "UP" if pred_dir == 1 else "DOWN",
            #     "confidence": confidence
            # }
        except Exception as e:
            return {"error": str(e)}

 #--------------------------------------------------------------------------   
 # This is anohter method of the base class implemented in this subclass  
    def get_results(self):
        return {
            "dates": self.dates,
            "actual_prices": self.actual_prices,
            "predicted_prices": self.predicted_prices,
            "true_directions": self.true_directions,
            "predicted_directions": self.predicted_directions,
            "confidence_scores": self.confidence_scores,
            "classification_accuracy": self.classification_metric.get(),
            "regression_mae": self.regression_metric.get()
        }
 #--------------------------------------------------------------------------   
  #--------------------------------------------------------------------------   

    def display_results(self):
        self.results = self.get_results()
        if not self.results:
            print("No results available.")
            return

        print(f"\n📊 Classification Accuracy: {self.results['classification_accuracy']:.4f}")
        print(f"📉 Regression MAE: {self.results['regression_mae']:.4f}")

        # Plot Predictions
        self._plot_predictions(
            self.results["dates"],
            self.results["actual_prices"],
            self.results["predicted_prices"]
        )

        # Show Prices
        self._show_prices(
            self.results["dates"],
            self.results["actual_prices"],
            self.results["predicted_prices"]
        )

        # Prediction Directions
        self._show_predictions(
            self.results["dates"],
            self.results["true_directions"],
            self.results["predicted_directions"],
            self.results["confidence_scores"]
        )

    def _plot_predictions(self, dates, actual_prices, predicted_prices):
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual_prices, label="Actual Price", color='blue', marker='o')
        plt.plot(dates, predicted_prices, label="Predicted Price", color='orange', linestyle='--', marker='x')
        plt.title("Actual vs Predicted Close Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _show_prices(self, dates, actual_prices, predicted_prices):
        for d, a, p in zip(dates, actual_prices, predicted_prices):
            print(f"Date: {d} => Actual Price: {a}, Predicted Price: {p}, Difference: {p-a}")

    def _show_predictions(self, dates, true_directions, predicted_directions, confidence_scores):
        for d, td, pd, cs in zip(dates, true_directions, predicted_directions, confidence_scores):
            print(f"Date: {d} => True : {td}, Predicted : {pd}, Confidence: {cs}")
        

#----------------------------------------------------------------------------   


#-----------------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-----------------------------------------------------------------------------
# This is another strategy
# Strategy with Technical Indicators and Warmup (Indicator+Model)
class IntradayMultiTaskModelwithTechnicalIndicatorsStrategy:
    def __init__(self, drift_confidence: float = 0.002, 
                 classifier_type: str = "logistic", 
                 regressor_type: str = "linear"):
        super().__init__()         

        self.classifier_pipeline = lambda: compose.Pipeline(
            preprocessing.StandardScaler(),
            self._get_classifier(classifier_type)
        )
        self.classifier = self.classifier_pipeline()

        self.regressor_pipeline = lambda: compose.Pipeline(
            preprocessing.StandardScaler(),
            self._get_regressor(regressor_type)
        )
        self.regressor = self.regressor_pipeline()

        self.classifier_drift_detector = drift.ADWIN(delta=drift_confidence)
        self.regressor_drift_detector = drift.ADWIN(delta=drift_confidence)

        self.classification_metric = metrics.Accuracy()
        self.regression_metric = metrics.MAE()

        self.dates = []
        self.actual_prices = []
        self.predicted_prices = []
        self.true_directions = []
        self.predicted_directions = []
        self.confidence_scores = []

        self.df_stream = pd.DataFrame()  # DataFrame to hold streaming data
        self.indicator_warmup_count = 0
        self.model_warmup_count = 0

    def _get_classifier(self, name):
        return {
            "logistic": LogisticRegression(),
            "hoeffding_tree": HoeffdingTreeClassifier(),
            "ar_forest": ARFClassifier(),
            "knn": KNNClassifier(n_neighbors=10),
            "naive_bayes": GaussianNB()
        }.get(name, LogisticRegression())

    def _get_regressor(self, name):
        return {
            "linear":LinearRegression(),
            "hoeffding_tree": HoeffdingTreeRegressor(),
            "ar_forest": ARFRegressor(),
            "knn": KNNRegressor(n_neighbors=10)
        }.get(name, LinearRegression())

    #-------------------------------------------------------------------------------

    def train(self, x1: dict, x2: dict, y_direction: int, y_log_return: float,   timestamp: datetime):
        # Predict before training (for drift check)
        pred_dir = self.classifier.predict_one(x1)
        pred_delta = self.regressor.predict_one(x2)

        # Update drift detectors
        self.classifier_drift_detector.update(int(pred_dir == y_direction))
        self.regressor_drift_detector.update(abs(pred_delta - y_log_return))

        # Check for drift and reset if needed
        if self.classifier_drift_detector.drift_detected:
            self.classifier = self.classifier_pipeline()
            print(f"[{timestamp}]📉 Drift detected in classifier. Resetting model.")

        if self.regressor_drift_detector.drift_detected:
            self.regressor = self.regressor_pipeline()
            print(f"[{timestamp}]📉 Drift detected in regressor. Resetting model.")

        # Learn
        self.classifier.learn_one(x1, y_direction)
        self.regressor.learn_one(x2, y_log_return)

        # Update metrics
        self.classification_metric.update(y_direction, pred_dir)
        self.regression_metric.update(y_log_return, pred_delta)

    def predict(self, x1: dict, x2: dict, current_price: float):
        direction = self.classifier.predict_one(x1)
        probas = self.classifier.predict_proba_one(x1)
        confidence_raw = probas.get(direction, 0.0) if probas else 0.0

        if confidence_raw is None or (isinstance(confidence_raw, float) and (math.isnan(confidence_raw) or confidence_raw < 0 or confidence_raw > 1)):
            confidence = 0.0
        else:
            confidence = confidence_raw

        log_return = self.regressor.predict_one(x2)
        if log_return is None or (isinstance(log_return, float) and math.isnan(log_return)):
            log_return = 0.0

        predicted_price = float(self._estimate_next_price(current_price, log_return))

        return {
            "direction": direction,
            "confidence": confidence,
            "log_return": log_return,
            "predicted_price": predicted_price
        }
    
    def _estimate_next_price(self, current_price: float, log_return: float) -> float:
        try:
            return current_price * math.exp(log_return)
        except:
            return current_price  # Fallback

    def get_metrics(self):
        return {
            "classification_accuracy": self.classification_metric.get(),
            "regression_mae": self.regression_metric.get()
        }

    #-------------------------------------------------------------------------------
    # Default window size is 10.
    # First 10 data points will be used for computing the RSI
    # Commpute technical indicators
    def _compute_technical_indicators(self, df: pd.DataFrame, window: int = 10):
        """Compute RSI, MACD, ATR, Fibonacci Retracement for last window."""

        # RSI (Relative Strength Index)
        rsi = ta.momentum.RSIIndicator(close=df["Close"], window=window).rsi().iloc[-1]

        # MACD (Moving Average Convergence Divergence)
        macd_indicator = ta.trend.MACD(close=df["Close"])
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        macd_diff = macd_indicator.macd_diff().iloc[-1]

        # ATR (Average True Range)
        atr = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=window).average_true_range().iloc[-1]

        # Fibonacci Retracement levels (from recent window)
        high_ = df["High"].max()
        low_ = df["Low"].min()
        diff = high_ - low_
        fib_382 = high_ - 0.382 * diff
        fib_618 = high_ - 0.618 * diff

        # print("Tech indicators created......")

        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_diff": macd_diff,
            "atr": atr,
            "fib_382": fib_382,
            "fib_618": fib_618,
        }

    # Generic class to be called to compute all the features
    # This function will be called in a loop during the online training of the model 
    def _compute_features(self, df: pd.DataFrame, window: int = 10):
        """Generate feature dict for the latest row using previous row and technical indicators."""
        prev_row = df.iloc[-2]
        curr_row = df.iloc[-1]

        # Base features from two most recent prices (current price and the previous price)
        base_features = {
            "price_change": (curr_row["Close"] - prev_row["Close"]) / prev_row["Close"],
            "volume_change": (curr_row["Volume"] - prev_row["Volume"]) / (prev_row["Volume"] + 1e-8),
            "high_low_range": (curr_row["High"] - curr_row["Low"]) / curr_row["Open"],
            "close_open_diff": (curr_row["Close"] - curr_row["Open"]) / curr_row["Open"],
        }

        technical_features = self._compute_technical_indicators(df, window)

        # Combine base and technical indicators
        combined_features = {**base_features, **technical_features}

        # print("Combined features created......")

        return base_features, combined_features

    def _compute_log_return(self, prev_close, curr_close):
        # print("Computed log returns....")
        return math.log(curr_close / prev_close)

    #-------------------------------------------------------------------------------
    # This is one of the methods of the base class implemented in this subclass  
    def run_on_data(self, prev_row, curr_row, warmup: str ="on", indicator_skip_count: int=10, model_skip_count: int=5) -> dict:
        try:
           # Append current row to DataFrame
            row_data = {
                "Date": datetime.strptime(curr_row["Date"], "%d-%b-%y"),
                "Open": float(curr_row["Open"]),
                "High": float(curr_row["High"]),
                "Low": float(curr_row["Low"]),
                "Close": float(curr_row["Close"]),
                "Volume": float(curr_row["Volume"]),
            }
            self.df_stream = pd.concat([self.df_stream, pd.DataFrame([row_data])], ignore_index=True)

            if warmup == "on" and self.indicator_warmup_count < indicator_skip_count:
                self.indicator_warmup_count = self.indicator_warmup_count + 1
                print(f"[Indicator Warmup Mode Skipping row {self.indicator_warmup_count}/{indicator_skip_count}")
                # print(f"{row_data['Date']}")
                return


            x1, x2 = self._compute_features(self.df_stream.iloc[-(indicator_skip_count+1):], window=indicator_skip_count)
            y_direction = int(float(curr_row["Close"]) > float(prev_row["Close"]))
            y_log_return = self._compute_log_return(float(prev_row["Close"]), float(curr_row["Close"]))
            current_price = float(prev_row["Close"])
            # timestamp = datetime.strptime(curr_row["Date"], "%d-%b-%y")
            timestamp = row_data["Date"]


            # Prediction (for drift and evaluation)
            pred_dir = self.classifier.predict_one(x1)
            confidence = round(self.classifier.predict_proba_one(x1).get(pred_dir, 0), 4)
            pred_return = self.regressor.predict_one(x2)
            if pred_return is None or (isinstance(pred_return, float) and math.isnan(pred_return)):
                pred_return = 0.0


            # Drift detection
            self.classifier_drift_detector.update(int(pred_dir == y_direction))
            self.regressor_drift_detector.update(abs(pred_return - y_log_return))
            # Reset model if drift is detected
            if self.classifier_drift_detector.drift_detected:
                self.classifier = self.classifier_pipeline()
                print(f"[{timestamp}] 📉 Classifier drift detected. Resetting model.")
            if self.regressor_drift_detector.drift_detected:
                self.regressor = self.regressor_pipeline()
                print(f"[{timestamp}] 📉 Regressor drift detected. Resetting model.")

            # Train - including model training during model warmup
            self.classifier.learn_one(x1, y_direction)
            self.regressor.learn_one(x2, y_log_return)

            # Let the model train from some initial samples (indicator_skip_count+1 TH sample to indicator_skip_count+model_skip_count TH sample)
            if warmup == "on" and self.model_warmup_count < model_skip_count:
                self.model_warmup_count = self.model_warmup_count + 1
                print(f"[Model Warmup Mode] Skipping row {self.model_warmup_count}/{model_skip_count}")
                # print(f"{row_data['Date']}")
                return

            # Start accumulating model performance metrics after the (1st + indicator_skip_count+ model_skip_count) number of samples
            # Update metrics
            self.classification_metric.update(y_direction, pred_dir)
            self.regression_metric.update(y_log_return, pred_return)

            # Start accumulating results after the (1st + indicator_skip_count+ model_skip_count) number of samples (e.g. 1+10+5=16)
            # So, we start accumulating from the 17th data point/sample
            # Save for plotting
            timestamp = datetime.strptime(curr_row["Date"], "%d-%b-%y")
            self.dates.append(timestamp)
            actual_price = float(curr_row["Close"])
            predicted_price = self._estimate_next_price(current_price, pred_return)
            self.actual_prices.append(actual_price)
            self.predicted_prices.append(predicted_price)
            self.true_directions.append("Up" if y_direction else "Down")
            self.predicted_directions.append("Up" if pred_dir else "Down")
            self.confidence_scores.append(confidence)


        except Exception as e:
            print("An error occurred: ", str(e))
            traceback.print_exc()

        # return {
        #         "timestamp": timestamp,
        #         "actual_price": actual_price,
        #         "predicted_price": predicted_price,
        #         "log_return": round(pred_return, 4),
        #         "direction": "UP" if pred_dir == 1 else "DOWN",
        #         "confidence": confidence
        #     }

 #--------------------------------------------------------------------------   
     # This is another method of the base class implemented in this subclass  
    def get_results(self):
        return {
            "dates": self.dates,
            "actual_prices": self.actual_prices,
            "predicted_prices": self.predicted_prices,
            "true_directions": self.true_directions,
            "predicted_directions": self.predicted_directions,
            "confidence_scores": self.confidence_scores,
            "classification_accuracy": self.classification_metric.get(),
            "regression_mae": self.regression_metric.get()
        }
 #--------------------------------------------------------------------------   

    def display_results(self):
        self.results = self.get_results()
        if not self.results:
            print("No results available.")
            return

        print(f"\n📊 Classification Accuracy: {self.results['classification_accuracy']:.4f}")
        print(f"📉 Regression MAE: {self.results['regression_mae']:.4f}")

        # Plot Predictions
        self._plot_predictions(
            self.results["dates"],
            self.results["actual_prices"],
            self.results["predicted_prices"]
        )

        # Show Prices
        self._show_prices(
            self.results["dates"],
            self.results["actual_prices"],
            self.results["predicted_prices"]
        )

        # Prediction Directions
        self._show_predictions(
            self.results["dates"],
            self.results["true_directions"],
            self.results["predicted_directions"],
            self.results["confidence_scores"]
        )

    def _plot_predictions(self, dates, actual_prices, predicted_prices):
        plt.figure(figsize=(12, 6))
        plt.plot(dates, actual_prices, label="Actual Price", color='blue', marker='o')
        plt.plot(dates, predicted_prices, label="Predicted Price", color='orange', linestyle='--', marker='x')
        plt.title("Actual vs Predicted Close Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _show_prices(self, dates, actual_prices, predicted_prices):
        for d, a, p in zip(dates, actual_prices, predicted_prices):
            print(f"Date: {d} => Actual Price: {a}, Predicted Price: {p}, Difference: {p-a}")

    def _show_predictions(self, dates, true_directions, predicted_directions, confidence_scores):
        for d, td, pd, cs in zip(dates, true_directions, predicted_directions, confidence_scores):
            print(f"Date: {d} => True : {td}, Predicted : {pd}, Confidence: {cs}")
        

#----------------------------------------------------------------------------   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------------------------------------------------------   

from river import anomaly, forest

class RiverAnomalyBoostedClassifierStrategy(Strategy):
    def __init__(self):
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            forest.ARFClassifier()
        )
        super().__init__()         
        self.anomaly_detector = anomaly.HalfSpaceTrees(seed=42)
        self.metric = metrics.Accuracy()
        self.predictions, self.timestamps = [], []

    def _compute_features(self, row):
        return {
            "close": float(row["Close"]),
            "open": float(row["Open"]),
            "volume": float(row["Volume"]),
            "range": float(row["High"]) - float(row["Low"]),
        }

    def run_on_data(self, prev_row, curr_row, **kwargs):
        x = self._compute_features(curr_row)
        anomaly_score = self.anomaly_detector.score_one(x)
        x["anomaly_score"] = anomaly_score
        direction = int(float(curr_row["Close"]) > float(prev_row["Close"]))
        pred = self.model.predict_one(x)
        self.metric.update(direction, pred)
        self.model.learn_one(x, direction)
        self.anomaly_detector.learn_one(x)
        self.predictions.append(pred)
        self.timestamps.append(curr_row["Date"])

    def get_results(self):
        return {
            "predicted_directions": self.predictions,
            "timestamps": self.timestamps,
            "accuracy": self.metric.get()
        }

    def display_results(self):
        self.results = self.get_results()
        if not self.results:
            print("No results available.")
            return

        print(f" Timestamp: {self.results['timestamps']}")
        print(f"\n📊 Predicted Directions: {self.results['predicted_directions']}")
        print(f"\n📊 Accuracy: {self.results['accuracy']:.4f}")


#----------------------------------------------------------------------------   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------------------------------------------------------   

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from collections import deque

class RNNPricePredictorStrategy(Strategy):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = self._build_model()
        self.memory = deque(maxlen=sequence_length)
        self.prices, self.predictions, self.timestamps = [], [], []

    def _build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, 1)),
            Dense(32, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation="linear")
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def run_on_data(self, prev_row, curr_row, **kwargs):
        close_price = float(curr_row["Close"])
        self.memory.append(close_price)

        if len(self.memory) == self.sequence_length:
            sequence = np.array(self.memory).reshape(1, self.sequence_length, 1)
            pred = self.model.predict(sequence, verbose=0)[0][0]
            self.predictions.append(pred)
            self.prices.append(close_price)
            self.timestamps.append(curr_row["Date"])

        warmup = kwargs.get("warmup", 10)

        # Train model only if enough data
        if len(self.prices) > self.sequence_length + warmup:
            X, y = [], []
            for i in range(len(self.prices) - self.sequence_length):
                X.append(self.prices[i:i+self.sequence_length])
                y.append(self.prices[i+self.sequence_length])
            X = np.array(X).reshape(-1, self.sequence_length, 1)
            y = np.array(y).reshape(-1, 1)
            self.model.fit(X, y, epochs=1, verbose=0)

    def get_results(self):
        return {
            "actual_prices": self.prices,
            "predicted_prices": float(self.predictions),
            "timestamps": self.timestamps
        }

    def display_results(self):
        self.results = self.get_results()
        if not self.results:
            print("No results available.")
            return

        print(f" Timestamp: {self.results['timestamps']}")
        print(f"\n📊 Predicted Prices: {self.results['predicted_prices']}")
        print(f"\n📊 Actual Prices: {self.results['actual_prices']}")
    
#----------------------------------------------------------------------------   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------------------------------------------------------   
class OnlineMomentumReversalStrategy(Strategy):
    def __init__(self):
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(),
            HoeffdingTreeClassifier()
        )
        self.rsi_window = deque(maxlen=14)
        self.dates, self.actions = [], []
        self.metric = metrics.Accuracy()

    def _compute_features(self, row):
        close = float(row["Close"])
        open_ = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        volume = float(row["Volume"])
        return {
            "close_open": (close - open_) / open_,
            "hl_range": (high - low) / open_,
            "rsi": self._compute_rsi(close),
            "volume": volume
        }

    def _compute_rsi(self, close):
        self.rsi_window.append(close)
        if len(self.rsi_window) < 14:
            return 50.0  # neutral
        gains = [max(0, self.rsi_window[i+1] - self.rsi_window[i]) for i in range(13)]
        losses = [max(0, self.rsi_window[i] - self.rsi_window[i+1]) for i in range(13)]
        avg_gain, avg_loss = sum(gains)/14, sum(losses)/14
        rs = avg_gain / (avg_loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def run_on_data(self, prev_row, curr_row, **kwargs):
        x = self._compute_features(curr_row)
        direction = int(float(curr_row["Close"]) > float(curr_row["Open"]))
        pred = self.model.predict_one(x)
        self.metric.update(direction, pred)
        self.model.learn_one(x, direction)
        self.dates.append(curr_row["Date"])
        self.actions.append("BUY" if pred == 1 else "SELL")

    def get_results(self):
        return {
            "dates": self.dates,
            "actions": self.actions,
            "accuracy": self.metric.get()
        }

    def display_results(self):
        self.results = self.get_results()
        if not self.results:
            print("No results available.")
            return

        print(f" Dates: {self.results['dates']}")
        print(f"\n Actions: {self.results['actions']}")
        print(f"\n Accuracy: {self.results['accuracy']:.4f}")
    



#----------------------------------------------------------------------------   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------------------------------------------------------   



#----------------------------------------------------------------------------   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------------------------------------------------------   
