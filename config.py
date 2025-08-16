import os
import logging
from typing import Dict, List
import numpy as np
from agent_utils import AgentConfig, EnvironmentConfig, TaskConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class for the agent
class Config:
    def __init__(self, agent_config: AgentConfig, env_config: EnvironmentConfig, task_configs: List[TaskConfig]):
        self.agent = agent_config
        self.environment = env_config
        self.tasks = task_configs
        self.validate()

    def validate(self):
        # Validate agent configuration
        if self.agent.num_actuators < 1:
            raise ValueError("Number of actuators must be at least 1.")
        if not np.isfinite(self.agent.mass).all():
            raise ValueError("Agent mass must be a finite number.")

        # Validate environment configuration
        if not self.environment.bounds or len(self.environment.bounds) != 2:
            raise ValueError("Environment bounds must be specified as a list of two 3D points.")
        if not np.isfinite(self.environment.gravity).all():
            raise ValueError("Gravity vector must be a finite 3D vector.")

        # Validate task configurations
        if not all(isinstance(task, TaskConfig) for task in self.tasks):
            raise TypeError("All task configurations must be instances of TaskConfig.")
        if any(task.priority is None or task.priority < 0 for task in self.tasks):
            raise ValueError("Task priorities must be non-negative integers.")

        logger.info("Configuration validated successfully.")

# Agent configuration class
class AgentConfig:
    def __init__(self, num_actuators: int, mass: np.ndarray):
        self.num_actuators = num_actuators
        self.mass = mass

# Environment configuration class
class EnvironmentConfig:
    def __init__(self, bounds: List[np.ndarray], gravity: np.ndarray):
        self.bounds = bounds
        self.gravity = gravity

# Task configuration class
class TaskConfig:
    def __init__(self, priority: int, objective: str, constraints: Dict[str, float] = None):
        self.priority = priority
        self.objective = objective
        self.constraints = constraints if constraints else {}

# Example usage
if __name__ == "__main__":
    # Example configurations
    agent_config = AgentConfig(num_actuators=6, mass=np.array([1.0, 0.5, 0.5]))
    env_config = EnvironmentConfig(bounds=[np.zeros(3), np.ones(3)], gravity=np.array([0, 0, -9.81]))
    task1 = TaskConfig(priority=1, objective="reach_target", constraints={"velocity": 0.8})
    task2 = TaskConfig(priority=2, objective="avoid_obstacle")
    task_configs = [task1, task2]

    # Create and validate configuration
    config = Config(agent_config, env_config, task_configs)

    # Access configuration parameters
    num_actuators = config.agent.num_actuators
    mass = config.agent.mass
    bounds = config.environment.bounds
    gravity = config.environment.gravity
    task_priorities = [task.priority for task in config.tasks]
    task_objectives = [task.objective for task in config.tasks]
    task_constraints = {task.objective: task.constraints for task in config.tasks}

    # Print configuration summary
    logger.info("Configuration summary:")
    logger.info(f"Agent has {num_actuators} actuators and mass: {mass}")
    logger.info(f"Environment bounds: {bounds}")
    logger.info(f"Gravity: {gravity}")
    logger.info(f"Task priorities: {task_priorities}")
    logger.info(f"Task objectives: {task_objectives}")
    logger.info(f"Task constraints: {task_constraints}")