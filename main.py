
#-------------------
# main.py
#-------------------


from algotrader import AlgoTrader
from river_strategy import (IntradayMultiTaskModelwithTechnicalIndicatorsStrategy, 
                            IntradayMultiTaskModelStrategy,
                            RiverAnomalyBoostedClassifierStrategy,
                            RNNPricePredictorStrategy,
                            OnlineMomentumReversalStrategy,
                          )
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
kwarg = {
          "warmup": 'on', 
          "indicator_skip_count": 10, 
          "model_skip_count": 5
        }

# Create a Strategy
# strategy = IntradayMultiTaskModelwithTechnicalIndicatorsStrategy(
#                     classifier_type="ar_forest", # 90.56%
#                     regressor_type="ar_forest" # MAE 0.0137 
#                 )

# strategy = IntradayMultiTaskModelStrategy(
#                     classifier_type="ar_forest", # 86.29%
#                     regressor_type="ar_forest" # MAE 0.0057 
#                 )

# strategy = IntradayMultiTaskModelStrategy(
#                     classifier_type="naive_bayes", # 75.8%
#                     regressor_type="ar_forest" # MAE 0.0058 
#                 )

# strategy = IntradayMultiTaskModelStrategy(
#                     classifier_type="ar_forest", # 89.5%
#                     regressor_type="hoeffding_tree" # MAE 0.0081 
#                 )

# strategy = IntradayMultiTaskModelStrategy(
#                     classifier_type="knn", # 78.6%
#                     regressor_type="knn" # MAE 0.0102 
#                 )

# strategy = IntradayMultiTaskModelStrategy(
#                     classifier_type="linear", # 81.8%
#                     regressor_type="logistic" # MAE 0.0081 
#                 )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
kwarg = {
         "warmup": 20
        }
strategy = RNNPricePredictorStrategy(sequence_length=10)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
kwarg ={}
# strategy = RiverAnomalyBoostedClassifierStrategy()
# strategy = OnlineMomentumReversalStrategy()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#------------------------------------------------------------
# Create an AlgoTrader object
ag = AlgoTrader("RILL.csv")
#----------------------------------------------------------------------------------
# Call the backtest method by passing the strategy and the warmup on/off indicator
ag.backtest(strategy, **kwarg)
#----------------------------------------------------------------------------------
# Display results of backtesting
ag.backtest_results()
#----------------------------------------------------------------------------------
