#-------------------
# algotrader.py
#-------------------

import csv

# Define the main Orchestrator AlgoTrader Class
class AlgoTrader:
      def __init__(self, csv_file_path):
        self.csv_path = csv_file_path
        self.strategy = None
        self.results = {}

      def backtest(self, strategy, **kwarg):
        self.strategy = strategy
        print("\nBacktesting ..........")

        with open(self.csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            reader.fieldnames = [name.lstrip('ï»¿') for name in reader.fieldnames]
            prev = next(reader)
            for curr in reader:
              _ = self.strategy.run_on_data(prev, curr, **kwarg)

              prev = curr
        # self.results = self.strategy.get_results()

      def backtest_results(self):
        self.strategy.display_results()

      # def backtest_results(self):
      #   if not self.results:
      #       print("No results available.")
      #       return

      #   print(f"\n📊 Classification Accuracy: {self.results['classification_accuracy']:.4f}")
      #   print(f"📉 Regression MAE: {self.results['regression_mae']:.4f}")

      #   # Plot Predictions
      #   self._plot_predictions(
      #       self.results["dates"],
      #       self.results["actual_prices"],
      #       self.results["predicted_prices"]
      #   )

      #   # Show Prices
      #   self._show_prices(
      #       self.results["dates"],
      #       self.results["actual_prices"],
      #       self.results["predicted_prices"]
      #   )

      #   # Prediction Directions
      #   self._show_predictions(
      #       self.results["dates"],
      #       self.results["true_directions"],
      #       self.results["predicted_directions"],
      #       self.results["confidence_scores"]
      #   )

      # def _plot_predictions(self, dates, actual_prices, predicted_prices):
      #   plt.figure(figsize=(12, 6))
      #   plt.plot(dates, actual_prices, label="Actual Price", color='blue', marker='o')
      #   plt.plot(dates, predicted_prices, label="Predicted Price", color='orange', linestyle='--', marker='x')
      #   plt.title("Actual vs Predicted Close Price")
      #   plt.xlabel("Date")
      #   plt.ylabel("Price")
      #   plt.xticks(rotation=45)
      #   plt.legend()
      #   plt.grid(True)
      #   plt.tight_layout()
      #   plt.show()

      # def _show_prices(self, dates, actual_prices, predicted_prices):
      #   for d, a, p in zip(dates, actual_prices, predicted_prices):
      #       print(f"Date: {d} => Actual Price: {a}, Predicted Price: {p}, Difference: {p-a}")

      # def _show_predictions(self, dates, true_directions, predicted_directions, confidence_scores):
      #   for d, td, pd, cs in zip(dates, true_directions, predicted_directions, confidence_scores):
      #       print(f"Date: {d} => True : {td}, Predicted : {pd}, Confidence: {cs}")
        