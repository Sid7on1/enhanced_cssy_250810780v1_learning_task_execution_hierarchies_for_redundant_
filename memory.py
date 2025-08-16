import logging
import numpy as np
import torch
from typing import List, Tuple, Dict
from collections import deque
from threading import Lock

# Define constants and configuration
MAX_MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
ALPHA = 0.6
BETA = 0.4
EPSILON = 0.001

# Define exception classes
class MemoryException(Exception):
    """Base exception class for memory-related errors"""
    pass

class MemoryFullException(MemoryException):
    """Exception raised when memory is full"""
    pass

class MemoryEmptyException(MemoryException):
    """Exception raised when memory is empty"""
    pass

# Define data structures and models
class Experience:
    """Represents a single experience in the memory"""
    def __init__(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class Memory:
    """Experience replay and memory class"""
    def __init__(self, max_size: int = MAX_MEMORY_SIZE, batch_size: int = BATCH_SIZE):
        self.max_size = max_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_size)
        self.lock = Lock()

    def add_experience(self, experience: Experience):
        """Add a new experience to the memory"""
        with self.lock:
            if len(self.memory) >= self.max_size:
                raise MemoryFullException("Memory is full")
            self.memory.append(experience)

    def sample_experiences(self) -> List[Experience]:
        """Sample a batch of experiences from the memory"""
        with self.lock:
            if len(self.memory) < self.batch_size:
                raise MemoryEmptyException("Memory is empty")
            indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
            return [self.memory[i] for i in indices]

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update the priorities of the experiences in the memory"""
        with self.lock:
            for i, priority in zip(indices, priorities):
                self.memory[i] = Experience(
                    state=self.memory[i].state,
                    action=self.memory[i].action,
                    reward=self.memory[i].reward,
                    next_state=self.memory[i].next_state,
                    done=self.memory[i].done
                )

    def calculate_priorities(self, experiences: List[Experience]) -> List[float]:
        """Calculate the priorities of the experiences"""
        priorities = []
        for experience in experiences:
            priority = (experience.reward + GAMMA * np.max(experience.next_state)) - experience.state
            priorities.append(priority)
        return priorities

    def get_size(self) -> int:
        """Get the current size of the memory"""
        with self.lock:
            return len(self.memory)

class PrioritizedMemory(Memory):
    """Prioritized experience replay and memory class"""
    def __init__(self, max_size: int = MAX_MEMORY_SIZE, batch_size: int = BATCH_SIZE, alpha: float = ALPHA, beta: float = BETA):
        super().__init__(max_size, batch_size)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=max_size)

    def add_experience(self, experience: Experience):
        """Add a new experience to the memory"""
        with self.lock:
            if len(self.memory) >= self.max_size:
                raise MemoryFullException("Memory is full")
            self.memory.append(experience)
            self.priorities.append(self.calculate_priority(experience))

    def sample_experiences(self) -> Tuple[List[Experience], List[int], List[float]]:
        """Sample a batch of experiences from the memory"""
        with self.lock:
            if len(self.memory) < self.batch_size:
                raise MemoryEmptyException("Memory is empty")
            indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=self.get_probabilities())
            experiences = [self.memory[i] for i in indices]
            priorities = [self.priorities[i] for i in indices]
            return experiences, indices, priorities

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update the priorities of the experiences in the memory"""
        with self.lock:
            for i, priority in zip(indices, priorities):
                self.priorities[i] = priority

    def calculate_priority(self, experience: Experience) -> float:
        """Calculate the priority of an experience"""
        priority = (experience.reward + GAMMA * np.max(experience.next_state)) - experience.state
        return priority ** self.alpha

    def get_probabilities(self) -> np.ndarray:
        """Get the probabilities of the experiences in the memory"""
        probabilities = np.array(self.priorities) / sum(self.priorities)
        return probabilities

def main():
    # Create a memory object
    memory = Memory()

    # Add some experiences to the memory
    for i in range(10):
        experience = Experience(
            state=np.random.rand(10),
            action=i,
            reward=np.random.rand(),
            next_state=np.random.rand(10),
            done=False
        )
        memory.add_experience(experience)

    # Sample some experiences from the memory
    experiences = memory.sample_experiences()
    for experience in experiences:
        print(experience.state, experience.action, experience.reward, experience.next_state, experience.done)

if __name__ == "__main__":
    main()