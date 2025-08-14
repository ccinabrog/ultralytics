# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
DXNN Utilities Module

This module provides utility functions for DXNN operations including
model validation, performance profiling, and device management.
"""

import json
import numpy as np
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ultralytics.utils import LOGGER


class DXNNUtils:
    """
    DXNN Utilities for common operations.
    
    This class provides utility functions for DXNN model operations,
    performance profiling, and system information.
    """
    
    def __init__(self):
        """Initialize DXNN Utilities."""
        pass
    
    @staticmethod
    def validate_dxnn_model(model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a DXNN model file.
        
        Args:
            model_path (str | Path): Path to the DXNN model
            
        Returns:
            Dict[str, Any]: Validation results
        """
        model_path = Path(model_path)
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        try:
            # Check if file exists
            if not model_path.exists():
                validation_result["errors"].append(f"Model file not found: {model_path}")
                return validation_result
            
            # Check file format
            if model_path.is_file():
                # Single file format
                if model_path.suffix.lower() != '.dxnn':
                    validation_result["warnings"].append(f"Unexpected file extension: {model_path.suffix}")
                
                # Try to read as JSON
                try:
                    with open(model_path, 'r') as f:
                        model_data = json.load(f)
                    
                    # Validate required fields
                    required_fields = ['format', 'version', 'model_data']
                    for field in required_fields:
                        if field not in model_data:
                            validation_result["errors"].append(f"Missing required field: {field}")
                    
                    validation_result["info"]["model_data"] = model_data
                    
                except json.JSONDecodeError:
                    validation_result["errors"].append("Invalid JSON format")
                    
            elif model_path.is_dir():
                # Directory format
                dxnn_files = list(model_path.glob("*.dxnn"))
                if not dxnn_files:
                    validation_result["errors"].append("No .dxnn file found in directory")
                else:
                    validation_result["info"]["dxnn_files"] = [str(f) for f in dxnn_files]
                
                # Check for metadata
                metadata_files = list(model_path.glob("metadata.*"))
                if metadata_files:
                    validation_result["info"]["metadata_files"] = [str(f) for f in metadata_files]
                else:
                    validation_result["warnings"].append("No metadata file found")
            
            # If no errors, mark as valid
            if not validation_result["errors"]:
                validation_result["valid"] = True
                
        except Exception as e:
            validation_result["errors"].append(f"Validation failed: {e}")
        
        return validation_result
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information for DXNN optimization.
        
        Returns:
            Dict[str, Any]: System information
        """
        info = {
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count()
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            })
        
        return info
    
    @staticmethod
    def profile_inference_time(
        runtime,
        input_data: Union[np.ndarray, List[np.ndarray]],
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> Dict[str, float]:
        """
        Profile inference time for a DXNN model.
        
        Args:
            runtime: DXNN runtime instance
            input_data (np.ndarray | List[np.ndarray]): Input data for inference
            num_runs (int): Number of inference runs for profiling
            warmup_runs (int): Number of warmup runs
            
        Returns:
            Dict[str, float]: Timing statistics
        """
        timings = []
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                start_time = time.time()
                runtime.inference(input_data)
                end_time = time.time()
                # Don't record warmup timings
            except Exception as e:
                LOGGER.warning(f"Warmup run failed: {e}")
        
        # Actual profiling runs
        for i in range(num_runs):
            try:
                start_time = time.time()
                runtime.inference(input_data)
                end_time = time.time()
                timings.append(end_time - start_time)
            except Exception as e:
                LOGGER.warning(f"Profiling run {i} failed: {e}")
        
        if not timings:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0
            }
        
        timings = np.array(timings)
        return {
            "mean": float(np.mean(timings)),
            "std": float(np.std(timings)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
            "median": float(np.median(timings))
        }
    
    @staticmethod
    def benchmark_model(
        model_path: Union[str, Path],
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        num_runs: int = 100,
        device: str = "npu"
    ) -> Dict[str, Any]:
        """
        Benchmark a DXNN model performance.
        
        Args:
            model_path (str | Path): Path to the DXNN model
            input_shape (tuple): Input tensor shape
            num_runs (int): Number of benchmark runs
            device (str): Target device
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        from .runtime import DXNNRuntime
        
        benchmark_result = {
            "model_path": str(model_path),
            "input_shape": input_shape,
            "device": device,
            "timing_stats": {},
            "system_info": {},
            "success": False
        }
        
        try:
            # Load model
            runtime = DXNNRuntime(device="npu", verbose=False)
            runtime.load_model(model_path)
            runtime.initialize()
            
            # Generate dummy input
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # Profile inference time
            timing_stats = DXNNUtils.profile_inference_time(runtime, input_data, num_runs)
            benchmark_result["timing_stats"] = timing_stats
            
            # Get system info
            benchmark_result["system_info"] = DXNNUtils.get_system_info()
            benchmark_result["system_info"]["device_info"] = runtime.get_device_info()
            
            benchmark_result["success"] = True
            
        except Exception as e:
            benchmark_result["error"] = str(e)
            LOGGER.error(f"Benchmark failed: {e}")
        
        return benchmark_result
    
    @staticmethod
    def compare_models(
        model_paths: List[Union[str, Path]],
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple DXNN models.
        
        Args:
            model_paths (List[str | Path]): List of model paths to compare
            input_shape (tuple): Input tensor shape
            num_runs (int): Number of benchmark runs per model
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        comparison_result = {
            "models": {},
            "summary": {},
            "success": False
        }
        
        try:
            # Benchmark each model
            for model_path in model_paths:
                model_name = Path(model_path).stem
                benchmark = DXNNUtils.benchmark_model(
                    model_path, input_shape, num_runs
                )
                comparison_result["models"][model_name] = benchmark
            
            # Create summary
            if comparison_result["models"]:
                mean_times = []
                model_names = []
                
                for name, result in comparison_result["models"].items():
                    if result.get("success", False):
                        mean_times.append(result["timing_stats"]["mean"])
                        model_names.append(name)
                
                if mean_times:
                    fastest_idx = np.argmin(mean_times)
                    slowest_idx = np.argmax(mean_times)
                    
                    comparison_result["summary"] = {
                        "fastest_model": model_names[fastest_idx],
                        "slowest_model": model_names[slowest_idx],
                        "fastest_time": mean_times[fastest_idx],
                        "slowest_time": mean_times[slowest_idx],
                        "speedup_factor": mean_times[slowest_idx] / mean_times[fastest_idx]
                    }
            
            comparison_result["success"] = True
            
        except Exception as e:
            comparison_result["error"] = str(e)
            LOGGER.error(f"Model comparison failed: {e}")
        
        return comparison_result
    
    @staticmethod
    def optimize_model_config(
        model_path: Union[str, Path],
        target_device: str = "npu",
        optimization_goals: List[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize model configuration for target device.
        
        Args:
            model_path (str | Path): Path to the DXNN model
            target_device (str): Target device for optimization
            optimization_goals (List[str]): Optimization goals (speed/memory/accuracy)
            
        Returns:
            Dict[str, Any]: Optimization recommendations
        """
        if optimization_goals is None:
            optimization_goals = ["speed"]
        
        optimization_result = {
            "model_path": str(model_path),
            "target_device": target_device,
            "optimization_goals": optimization_goals,
            "recommendations": {},
            "success": False
        }
        
        try:
            # Get system info
            system_info = DXNNUtils.get_system_info()
            
            # Generate recommendations based on NPU and goals
            recommendations = {}
            
            if "speed" in optimization_goals:
                recommendations["device"] = "npu"
                recommendations["batch_size"] = 4  # Optimize for NPU
                recommendations["optimization_level"] = "fast"
            
            if "memory" in optimization_goals:
                recommendations["batch_size"] = 1
                recommendations["optimization_level"] = "fast"
                recommendations["memory_pool"] = "minimal"
            
            if "accuracy" in optimization_goals:
                recommendations["optimization_level"] = "accurate"
                recommendations["precision"] = "fp32"
            
            optimization_result["recommendations"] = recommendations
            optimization_result["system_info"] = system_info
            optimization_result["success"] = True
            
        except Exception as e:
            optimization_result["error"] = str(e)
            LOGGER.error(f"Optimization failed: {e}")
        
        return optimization_result
    
    @staticmethod
    def export_benchmark_report(
        benchmark_results: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Export benchmark results to a report file.
        
        Args:
            benchmark_results (Dict[str, Any]): Benchmark results
            output_path (str | Path): Path to save the report
            
        Returns:
            bool: True if export successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create report content
            report_content = f"""
# DXNN Model Benchmark Report

## Summary
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Model: {benchmark_results.get('model_path', 'Unknown')}
- Device: {benchmark_results.get('device', 'Unknown')}

## Timing Statistics
"""
            
            timing_stats = benchmark_results.get("timing_stats", {})
            for key, value in timing_stats.items():
                report_content += f"- {key}: {value:.4f}s\n"
            
            report_content += "\n## System Information\n"
            system_info = benchmark_results.get("system_info", {})
            for key, value in system_info.items():
                report_content += f"- {key}: {value}\n"
            
            # Save report
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to export benchmark report: {e}")
            return False
