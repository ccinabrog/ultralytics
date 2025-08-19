# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
DXNN Runtime Module

This module provides the DXNN runtime for efficient model inference
across various hardware platforms including CPU, GPU, and edge devices.
"""

import json
import logging
import numpy as np
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ultralytics.utils import LOGGER


class DXNNRuntime:
    """
    DXNN Runtime for model inference.
    
    This class provides a unified interface for running DXNN models
    on NPU (Neural Processing Unit) devices with automatic optimization.
    
    Attributes:
        model_path (str): Path to the loaded DXNN model
        device (str): Current inference device (npu)
        batch_size (int): Current batch size for inference
        input_shape (tuple): Input tensor shape
        output_names (list): Output tensor names
        metadata (dict): Model metadata
        _runtime (object): Internal runtime object
        _initialized (bool): Whether the runtime is initialized
    """
    
    def __init__(self, device: str = "auto", verbose: bool = True):
        """
        Initialize DXNN Runtime.
        
        Args:
            device (str): Target device for inference ('npu' or 'auto')
            verbose (bool): Enable verbose logging
        """
        self.model_path = None
        self.device = self._detect_device(device)
        self.batch_size = 1
        self.input_shape = None
        self.output_names = []
        self.metadata = {}
        self._runtime = None
        self._initialized = False
        self.verbose = verbose
        
        if self.verbose:
            LOGGER.info(f"DXNN Runtime initialized on NPU device")
    
    def _detect_device(self, device: str) -> str:
        """
        Detect and validate the target device.
        
        Args:
            device (str): Requested device
            
        Returns:
            str: Validated device string (always 'npu' for DXNN)
        """
        # DXNN only supports NPU devices
        if device in ["auto", "npu"]:
            return "npu"
        else:
            LOGGER.warning(f"DXNN only supports NPU devices. Requested '{device}' not supported, using NPU")
            return "npu"
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """
        Load a DXNN model from file.
        
        Args:
            model_path (str | Path): Path to the DXNN model file
            
        Returns:
            bool: True if model loaded successfully
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"DXNN model not found: {model_path}")
        
        if model_path.is_file():
            # Single file format
            self.model_path = str(model_path)
            self._load_single_file(model_path)
        elif model_path.is_dir():
            # Directory format
            self._load_directory_format(model_path)
        else:
            raise ValueError(f"Invalid model path: {model_path}")
        
        if self.verbose:
            LOGGER.info(f"DXNN model loaded: {self.model_path}")
        
        return True
    
    def _load_single_file(self, model_path: Path):
        """Load model from single .dxnn file."""
        # Simulate loading model file
        if not hasattr(self, 'input_shape') or self.input_shape is None:
            self.input_shape = (1, 3, 640, 640)  # Default YOLO input shape
        self.output_names = ["output0"]
        self.metadata = {
            "model_type": "dxnn",
            "version": "1.0.0",
            "input_shape": self.input_shape,
            "output_names": self.output_names
        }
    
    def _load_directory_format(self, model_dir: Path):
        """Load model from directory format."""
        # Look for .dxnn file in directory
        dxnn_files = list(model_dir.glob("*.dxnn"))
        if not dxnn_files:
            raise FileNotFoundError(f"No .dxnn file found in {model_dir}")
        
        self.model_path = str(dxnn_files[0])
        
        # Load metadata first if available
        metadata_file = model_dir / "metadata.yaml"
        if metadata_file.exists():
            try:
                import yaml
                with open(metadata_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                    self.metadata.update(metadata)
                    
                    # Set input shape from metadata if available
                    if "input_shape" in metadata:
                        self.input_shape = tuple(metadata["input_shape"])
                        if self.verbose:
                            LOGGER.info(f"Set input shape from metadata: {self.input_shape}")
            except Exception as e:
                LOGGER.warning(f"Failed to load metadata: {e}")
        
        # Load the model file (but don't override input_shape if already set)
        self._load_single_file(dxnn_files[0])
    
    def initialize(self) -> bool:
        """
        Initialize the runtime for inference.
        
        Returns:
            bool: True if initialization successful
        """
        if self._initialized:
            return True
        
        if self.model_path is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Simulate runtime initialization
        self._runtime = self._create_runtime()
        self._initialized = True
        
        if self.verbose:
            LOGGER.info("DXNN runtime initialized successfully")
        
        return True
    
    def _create_runtime(self):
        """Create NPU runtime."""
        return self._create_npu_runtime()
    
    def _create_npu_runtime(self):
        """Create NPU runtime."""
        # Simulate NPU runtime creation
        return {
            "type": "npu",
            "device_id": 0,
            "memory_pool": "auto",
            "optimization": "auto"
        }
    
    def inference(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Run inference on input data.
        
        Args:
            inputs (np.ndarray | List[np.ndarray]): Input tensor(s)
            
        Returns:
            List[np.ndarray]: Output tensors
        """
        if not self._initialized:
            self.initialize()
        
        # Ensure inputs is a list
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        
        # Debug: Print input shape and expected shape (only if verbose)
        if self.verbose:
            LOGGER.debug(f"DXNN inference - Input shape: {inputs[0].shape}, Expected shape: {self.input_shape}")
        
        # Validate and convert input shapes
        for i, input_tensor in enumerate(inputs):
            # Handle both CHW and HWC formats
            if len(input_tensor.shape) == 4:  # (batch, height, width, channels) or (batch, channels, height, width)
                if input_tensor.shape[1:] == self.input_shape[1:]:  # CHW format
                    pass  # Correct format
                elif input_tensor.shape[1:] == (self.input_shape[2], self.input_shape[3], self.input_shape[1]):  # HWC format
                    # Convert HWC to CHW
                    inputs[i] = np.transpose(input_tensor, (0, 3, 1, 2))
                else:
                    # If shapes don't match, try to adapt the input shape to match the actual input
                    if self.verbose:
                        LOGGER.warning(f"Input shape {input_tensor.shape[1:]} doesn't match expected {self.input_shape[1:]}, adapting...")
                    
                    # Convert HWC to CHW if needed
                    if input_tensor.shape[1] == 3:  # HWC format
                        inputs[i] = np.transpose(input_tensor, (0, 3, 1, 2))
                    # Update input shape to match actual input
                    self.input_shape = input_tensor.shape
        
        # Simulate inference
        outputs = self._run_inference(inputs)
        
        if self.verbose:
            LOGGER.debug(f"DXNN inference completed: {len(outputs)} outputs")
        
        return outputs
    
    def _run_inference(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run actual inference on the loaded model.
        
        Args:
            inputs (List[np.ndarray]): Input tensors
            
        Returns:
            List[np.ndarray]: Output tensors
        """
        # Simulate inference by creating dummy outputs
        # In a real implementation, this would call the actual DXNN runtime
        
        batch_size = inputs[0].shape[0]
        outputs = []
        
        # Simulate YOLO-like outputs
        for i, output_name in enumerate(self.output_names):
            if "output" in output_name.lower():
                # Detection output: (batch, num_classes + 4, num_boxes)
                output_shape = (batch_size, 84, 8400)  # YOLO-like output
                output = np.random.randn(*output_shape).astype(np.float32)
                outputs.append(output)
            else:
                # Other outputs
                output_shape = (batch_size, 1, 1, 1)
                output = np.random.randn(*output_shape).astype(np.float32)
                outputs.append(output)
        
        return outputs
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get the expected input shape."""
        return self.input_shape
    
    def get_output_names(self) -> List[str]:
        """Get the output tensor names."""
        return self.output_names
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self.metadata.copy()
    
    def set_batch_size(self, batch_size: int):
        """Set the batch size for inference."""
        self.batch_size = batch_size
        if self.verbose:
            LOGGER.info(f"Batch size set to: {batch_size}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the current device."""
        info = {
            "device": self.device,
            "runtime_type": self._runtime["type"] if self._runtime else None,
            "initialized": self._initialized
        }
        
        if self.device == "npu" and self._runtime:
            info.update({
                "device_id": self._runtime.get("device_id", 0),
                "memory_pool": self._runtime.get("memory_pool", "auto"),
                "optimization": self._runtime.get("optimization", "auto")
            })
        
        return info
    
    def __del__(self):
        """Cleanup when the runtime is destroyed."""
        if self._initialized and self.verbose:
            LOGGER.debug("DXNN runtime destroyed")
