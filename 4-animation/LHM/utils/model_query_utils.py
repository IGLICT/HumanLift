# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-16 13:58:28
# @Function      : Automatically select models based on available GPU memory.

import subprocess
import sys

try:
    import GPUtil
except ImportError:
    package_name = "GPUtil"
    print(f"{package_name} is not installed. Installing now...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
    print(f"{package_name} has been successfully installed.")
    import GPUtil 


class AutoModelSwitcher:
    """Automatically selects the most suitable model based on available GPU memory.
    
    Attributes:
        model_requirements (dict): Model names mapped to their minimum GPU requirements (MB)
        sorted_models (list): Models sorted by memory requirements (descending) and HF priority
        extra_memory (int): Additional memory buffer to reserve for other processes
        get_memory (callable): Function to check available GPU memory
        available_mb (float): Current available GPU memory in MB
    """
    
    def __init__(self, model_requirements, get_memory_func=None, extra_memory=0):
        """Initialize the model switcher.
        
        Args:
            model_requirements (dict): {model_name: min_required_memory(MB)}
            get_memory_func (callable, optional): Custom function to check available GPU memory
            extra_memory (int, optional): Additional memory buffer to reserve (MB)
        """
        self.model_requirements = model_requirements
        # Sort models by: 1. Memory requirements (descending) 2. HF models first
        self.sorted_models = sorted(
            model_requirements.items(),
            key=lambda x: (-x[1], '-HF' not in x[0])
        )
        
        self.extra_memory = extra_memory
        
        # Initialize memory checking method
        if get_memory_func is None:
            self.get_memory = self._default_memory_check
        else:
            self.get_memory = get_memory_func
        
        self.available_mb = self._default_memory_check()

    def _default_memory_check(self, gpu_id=0):
        """Check available GPU memory using GPUtil.
        
        Args:
            gpu_id (int, optional): Target GPU device ID
            
        Returns:
            float: Available memory in MB
            
        Raises:
            RuntimeError: If no GPUs are found
            IndexError: If specified GPU ID is invalid
        """
        gpus = GPUtil.getGPUs()
        
        if not gpus:
            raise RuntimeError("No available GPUs detected")

        if gpu_id >= len(gpus):
            raise IndexError(f"Invalid GPU ID {gpu_id}. Only {len(gpus)} GPUs available")

        gpu = gpus[gpu_id]
        total_memory = gpu.memoryTotal
        used_memory = gpu.memoryUsed
        available_memory = total_memory - used_memory

        # Print memory status
        print(
            f"GPU {gpu.id}: "
            f"Total: {total_memory} MB, "
            f"Used: {used_memory} MB, "
            f"Available: {available_memory/1024:.2f} GB"
        )

        return available_memory

    def query(self, model_name):
        """Query the best available model for current memory conditions.
        
        Args:
            model_name (str): The preferred model name to try first
            
        Returns:
            str: Name of the selected model
            
        Raises:
            KeyError: If requested model is not in the configuration
            ValueError: If no suitable model can be found
        """
        if model_name not in self.model_requirements:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        
        available_mb = self.available_mb - self.extra_memory
        required_mb = self.model_requirements[model_name]
        
        # Return preferred model if sufficient memory
        if available_mb >= required_mb:
            return model_name
        
        # Find the best alternative model
        for name, req in self.sorted_models:
            if req <= available_mb:
                print(f"Insufficient memory, switching to largest available model: "
                      f"{model_name} -> {name}")
                return name
        
        raise ValueError("No available model meets current memory constraints")

    def get_available_models(self):
        """Get list of currently available models sorted by priority.
        
        Returns:
            list: Names of available models in priority order
        """
        available_mb = self.available_mb - self.extra_memory
        return [name for name, req in self.sorted_models if req <= available_mb]