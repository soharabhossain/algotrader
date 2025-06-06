
from abc import ABC, abstractmethod

# Strategy Base Class
# Each strategy class implemented by a develper needs to inherit from this base class implementing the two abstract methods defined here
class Strategy(ABC):
    @abstractmethod
    def run_on_data(self, prev, curr, warmup, model_skip_count, indicator_skip_count) -> dict:
        """Handles a new data tick from a live stream/or offline CSV and returns prediction + metrics info."""
        pass

    @abstractmethod
    def display_results(self) -> dict:
        """Returns final metrics after backtest."""
        pass

#-------------------------------------------------------------------------------------------------------------------
