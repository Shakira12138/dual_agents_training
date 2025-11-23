"""
Shared buffer for storing Agent A's conversation data for Agent B's offline training.
This replaces the database-based approach with a simpler in-memory buffer.
"""
import asyncio
import threading
from collections import deque
from typing import Dict, List, Optional
import copy

from slime.utils.types import Sample


class SharedTrainingBuffer:
    """
    Thread-safe shared buffer for storing training data from Agent A to Agent B.
    
    This buffer stores Sample objects that Agent B can directly use for training.
    """
    def __init__(self, max_size: int = 1000):
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.total_written = 0
        self.total_read = 0
    
    def add_sample(self, sample: Sample):
        """
        Add a Sample to the buffer for Agent B's training.
        
        Args:
            sample: Sample object containing conversation data from Agent A
        """
        with self.condition:
            # Deep copy to avoid reference issues
            self.buffer.append(copy.deepcopy(sample))
            self.total_written += 1
            self.condition.notify_all()
            print(f"[SharedBuffer] Added sample to buffer. Buffer size: {len(self.buffer)}, Total written: {self.total_written}")
    
    def get_sample(self, timeout: Optional[float] = None) -> Optional[Sample]:
        """
        Get a Sample from the buffer for Agent B's training.
        
        Args:
            timeout: Maximum time to wait for a sample (None = wait indefinitely)
            
        Returns:
            Sample object if available, None if timeout
        """
        with self.condition:
            if len(self.buffer) == 0:
                if timeout is None:
                    # Wait indefinitely
                    self.condition.wait()
                else:
                    # Wait with timeout
                    if not self.condition.wait(timeout=timeout):
                        return None
            
            if len(self.buffer) > 0:
                sample = self.buffer.popleft()
                self.total_read += 1
                print(f"[SharedBuffer] Retrieved sample from buffer. Buffer size: {len(self.buffer)}, Total read: {self.total_read}")
                return sample
            
            return None
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
            print("[SharedBuffer] Buffer cleared")


# Global shared buffer instance
_shared_buffer: Optional[SharedTrainingBuffer] = None
_buffer_lock = threading.Lock()


def get_shared_buffer() -> SharedTrainingBuffer:
    """
    Get or create the global shared buffer instance.
    
    Returns:
        SharedTrainingBuffer instance
    """
    global _shared_buffer
    with _buffer_lock:
        if _shared_buffer is None:
            _shared_buffer = SharedTrainingBuffer(max_size=1000)
        return _shared_buffer


def add_training_sample(sample: Sample):
    """
    Add a training sample to the shared buffer (called by Agent A).
    
    Args:
        sample: Sample object containing conversation data
    """
    buffer = get_shared_buffer()
    buffer.add_sample(sample)


def get_training_sample(timeout: Optional[float] = None) -> Optional[Sample]:
    """
    Get a training sample from the shared buffer (called by Agent B).
    
    Args:
        timeout: Maximum time to wait for a sample (None = wait indefinitely)
        
    Returns:
        Sample object if available, None if timeout
    """
    buffer = get_shared_buffer()
    return buffer.get_sample(timeout=timeout)

