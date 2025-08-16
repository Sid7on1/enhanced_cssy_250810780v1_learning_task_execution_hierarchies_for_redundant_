import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

# Define constants
VELOCITY_THRESHOLD = 0.5  # velocity threshold for flow theory
FLOW_THEORY_THRESHOLD = 0.8  # flow theory threshold

# Define configuration settings
class Configuration:
    def __init__(self, velocity_threshold: float = VELOCITY_THRESHOLD, flow_theory_threshold: float = FLOW_THEORY_THRESHOLD):
        """
        Configuration settings for agent evaluation metrics.

        Args:
        - velocity_threshold (float): velocity threshold for flow theory (default: 0.5)
        - flow_theory_threshold (float): flow theory threshold (default: 0.8)
        """
        self.velocity_threshold = velocity_threshold
        self.flow_theory_threshold = flow_theory_threshold

# Define exception classes
class EvaluationError(Exception):
    """Base class for evaluation-related exceptions."""
    pass

class InvalidInputError(EvaluationError):
    """Raised when invalid input is provided."""
    pass

class EvaluationMetric:
    def __init__(self, configuration: Configuration):
        """
        Base class for evaluation metrics.

        Args:
        - configuration (Configuration): configuration settings
        """
        self.configuration = configuration

    def calculate(self, data: np.ndarray) -> float:
        """
        Calculate the evaluation metric.

        Args:
        - data (np.ndarray): input data

        Returns:
        - float: calculated metric value
        """
        raise NotImplementedError

class VelocityThresholdMetric(EvaluationMetric):
    def calculate(self, data: np.ndarray) -> float:
        """
        Calculate the velocity threshold metric.

        Args:
        - data (np.ndarray): input data

        Returns:
        - float: calculated metric value
        """
        velocity = np.mean(data)
        if velocity > self.configuration.velocity_threshold:
            return 1.0
        else:
            return 0.0

class FlowTheoryMetric(EvaluationMetric):
    def calculate(self, data: np.ndarray) -> float:
        """
        Calculate the flow theory metric.

        Args:
        - data (np.ndarray): input data

        Returns:
        - float: calculated metric value
        """
        flow = np.mean(data)
        if flow > self.configuration.flow_theory_threshold:
            return 1.0
        else:
            return 0.0

class AgentEvaluator:
    def __init__(self, configuration: Configuration):
        """
        Agent evaluator class.

        Args:
        - configuration (Configuration): configuration settings
        """
        self.configuration = configuration
        self.metrics = [VelocityThresholdMetric(configuration), FlowTheoryMetric(configuration)]

    def evaluate(self, data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the agent using the specified metrics.

        Args:
        - data (np.ndarray): input data

        Returns:
        - Dict[str, float]: dictionary of metric values
        """
        metric_values = {}
        for metric in self.metrics:
            try:
                metric_value = metric.calculate(data)
                metric_values[type(metric).__name__] = metric_value
            except Exception as e:
                logging.error(f"Error calculating metric {type(metric).__name__}: {str(e)}")
        return metric_values

    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate the input data.

        Args:
        - data (np.ndarray): input data

        Returns:
        - bool: True if input is valid, False otherwise
        """
        if not isinstance(data, np.ndarray):
            raise InvalidInputError("Input must be a numpy array")
        if data.size == 0:
            raise InvalidInputError("Input array cannot be empty")
        return True

def main():
    # Create configuration settings
    configuration = Configuration()

    # Create agent evaluator
    evaluator = AgentEvaluator(configuration)

    # Generate sample data
    data = np.random.rand(100)

    # Validate input data
    try:
        evaluator.validate_input(data)
    except InvalidInputError as e:
        logging.error(f"Invalid input: {str(e)}")
        return

    # Evaluate agent
    metric_values = evaluator.evaluate(data)
    logging.info(f"Metric values: {metric_values}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()