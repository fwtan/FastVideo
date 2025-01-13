import time
import json
import torch
import torch.distributed as dist
import threading
import functools
from collections import defaultdict
from typing import Optional, Dict, List, Callable, Any, DefaultDict

class Timing:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation with thread safety"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Timing, cls).__new__(cls)
                cls._instance._init_attributes()
            return cls._instance

    def _init_attributes(self):
        """Initialize instance attributes"""
        self.timing_dict = defaultdict(list)
        self.start_times = {}
        self.stats: DefaultDict[str, Dict[str, float]] = defaultdict(
            lambda: {'total_time': 0.0, 'count': 0, 'avg_time': 0.0}
        )
        
    def start(self, block_name: str, gpu_rank: int):
        """Start timing a code block
        Args:
            block_name: identifier for the code block
            gpu_rank: GPU device rank
        """
        with self._lock:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_times[(block_name, gpu_rank)] = time.perf_counter()
        
    def end(self, block_name: str, gpu_rank: int):
        """End timing a code block
        Args:
            block_name: identifier for the code block
            gpu_rank: GPU device rank
        """
        with self._lock:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            if (block_name, gpu_rank) not in self.start_times:
                raise ValueError(f"No start time found for block {block_name} on GPU {gpu_rank}")
                
            start_time = self.start_times.pop((block_name, gpu_rank))
            self.timing_dict[block_name].append((start_time, end_time, gpu_rank))
        
    def _sync_across_gpus(self, local_time: float) -> float:
        """Synchronize timing across multiple GPUs and compute average"""
        if not dist.is_initialized():
            return local_time
            
        world_size = dist.get_world_size()
        times = torch.tensor([local_time], device='cuda')
        times_list = [torch.zeros_like(times) for _ in range(world_size)]
        dist.all_gather(times_list, times)
        
        avg_time = sum(t.item() for t in times_list) / world_size
        return avg_time
        
    def get_time(self, with_statistics: bool = True) -> str:
        """Get timing results for all blocks
        Args:
            with_statistics: If True, include call count and average time per call
        Returns:
            JSON string containing timing results and optional statistics
        """
        with self._lock:
            result = {}
            
            for block_name, time_list in self.timing_dict.items():
                # Calculate total time for each GPU per block
                gpu_times = defaultdict(float)
                for start, end, gpu_rank in time_list:
                    gpu_times[gpu_rank] += end - start
                    
                # Average times across GPUs
                avg_times = []
                for gpu_rank, total_time in gpu_times.items():
                    avg_times.append(self._sync_across_gpus(total_time))
                    
                # Compute final average across all GPUs
                total_time = sum(avg_times) / len(avg_times) if avg_times else 0
                
                # Update statistics
                self.stats[block_name]['total_time'] = total_time
                self.stats[block_name]['count'] = len(time_list) // len(gpu_times)
                self.stats[block_name]['avg_time'] = (
                    total_time / self.stats[block_name]['count'] 
                    if self.stats[block_name]['count'] > 0 else 0
                )
                
                if with_statistics:
                    result[block_name] = {
                        'total_time': total_time,
                        'call_count': self.stats[block_name]['count'],
                        'avg_time_per_call': self.stats[block_name]['avg_time']
                    }
                else:
                    result[block_name] = total_time
                
            return json.dumps(result, indent=2)
        
    def clean(self, block_name: Optional[str] = None):
        """Clean timing records for specified block or all blocks
        Args:
            block_name: if provided, clean only this block's records
                       if None, clean all blocks
        """
        with self._lock:
            if block_name is not None:
                if block_name in self.timing_dict:
                    self.timing_dict.pop(block_name)
                    self.stats.pop(block_name)
                    # Clean any pending start times
                    keys_to_remove = [
                        k for k in self.start_times.keys() 
                        if k[0] == block_name
                    ]
                    for k in keys_to_remove:
                        self.start_times.pop(k)
            else:
                self.timing_dict.clear()
                self.start_times.clear()
                self.stats.clear()
            
    def reset(self):
        """Reset all timing records (alias for clean())"""
        self.clean()

    def timer(self, func_name: Optional[str] = None) -> Callable:
        """Decorator for timing functions"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Get current GPU rank
                gpu_rank = 0
                if torch.cuda.is_available():
                    if dist.is_initialized():
                        gpu_rank = dist.get_rank()
                    else:
                        gpu_rank = torch.cuda.current_device()

                name = func_name or func.__qualname__
                self.start(name, gpu_rank)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end(name, gpu_rank)
            
            return wrapper
        return decorator

# Create a global timer instance
global_timer = Timing()

# Function to get the global timer instance
def get_timer() -> Timing:
    """Get the global timer instance"""
    return global_timer

if __name__ == "__main__":
    import torch.nn as nn
    import time

    # Test Case 1: Basic module with timed forward pass
    class SimpleTransformer(nn.Module):
        @global_timer.timer("transformer_forward")
        def forward(self, x):
            time.sleep(0.1)  # Simulate computation
            return x + 1

    # Test Case 2: Multiple blocks timing
    class ComplexModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleTransformer()

        @global_timer.timer("preprocess")
        def preprocess(self, x):
            time.sleep(0.05)  # Simulate preprocessing
            return x * 2

        @global_timer.timer("postprocess")
        def postprocess(self, x):
            time.sleep(0.03)  # Simulate postprocessing
            return x / 2

        def forward(self, x):
            x = self.preprocess(x)
            x = self.transformer(x)
            x = self.postprocess(x)
            return x

    def run_tests():
        print("Starting timing tests...")
        
        # Test 1: Single block timing
        print("\nTest 1: Single block timing")
        model = SimpleTransformer()
        input_data = torch.randn(32, 100, 512)
        
        # Run model multiple times
        for _ in range(3):
            output = model(input_data)
        
        results = global_timer.get_time()
        print("Single block results:")
        print(results)
        
        # Clean specific block
        global_timer.clean("transformer_forward")
        
        # Test 2: Multiple blocks timing
        print("\nTest 2: Multiple blocks timing")
        complex_model = ComplexModule()
        
        # Run model multiple times with different blocks
        for _ in range(5):
            output = complex_model(input_data)
        
        results = global_timer.get_time()
        print("Multiple blocks results:")
        print(results)
        
        # Test 3: Manual block timing
        print("\nTest 3: Manual block timing")
        global_timer.start("manual_block", 0)
        time.sleep(0.2)  # Simulate some work
        global_timer.end("manual_block", 0)
        
        results = global_timer.get_time()
        print("Results after manual timing:")
        print(results)
        
        # Test 4: Clean specific block
        print("\nTest 4: Clean specific block")
        global_timer.clean("preprocess")
        results = global_timer.get_time()
        print("Results after cleaning 'preprocess' block:")
        print(results)
        
        # Test 5: Clean all blocks
        print("\nTest 5: Clean all blocks")
        global_timer.clean()
        results = global_timer.get_time()
        print("Results after cleaning all blocks:")
        print(results)

    # Run all tests
    run_tests()