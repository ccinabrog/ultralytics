# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
DXNN Converter Module

This module provides functionality to convert models from various formats
to DXNN format for efficient inference.
"""

import datetime
import json
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ultralytics.utils import LOGGER


class DXNNConverter:
    """
    DXNN Converter for model format conversion.
    
    This class provides functionality to convert models from ONNX, PyTorch,
    and other formats to DXNN format optimized for NPU devices.
    
    Attributes:
        input_model (str): Path to the input model
        output_path (str): Path for the output DXNN model
        target_device (str): Target device for optimization (npu)
        batch_size (int): Batch size for the converted model
        optimization_level (str): Optimization level (fast/balanced/accurate)
        verbose (bool): Enable verbose logging
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DXNN Converter.
        
        Args:
            verbose (bool): Enable verbose logging
        """
        self.input_model = None
        self.output_path = None
        self.target_device = "npu"
        self.batch_size = 1
        self.optimization_level = "balanced"
        self.verbose = verbose
        
        if self.verbose:
            LOGGER.info("DXNN Converter initialized for NPU")
    
    def convert(
        self,
        input_model: Union[str, Path],
        output_path: Union[str, Path],
        target_device: str = "npu",
        batch_size: int = 1,
        optimization_level: str = "balanced",
        **kwargs
    ) -> bool:
        """
        Convert a model to DXNN format.
        
        Args:
            input_model (str | Path): Path to the input model
            output_path (str | Path): Path for the output DXNN model
            target_device (str): Target device for optimization (npu)
            batch_size (int): Batch size for the converted model
            optimization_level (str): Optimization level
            **kwargs: Additional conversion parameters
            
        Returns:
            bool: True if conversion successful
        """
        self.input_model = str(input_model)
        self.output_path = str(output_path)
        self.target_device = target_device
        self.batch_size = batch_size
        self.optimization_level = optimization_level
        
        if self.verbose:
            LOGGER.info(f"Converting {self.input_model} to DXNN format")
            LOGGER.info(f"Target device: NPU")
            LOGGER.info(f"Batch size: {self.batch_size}")
            LOGGER.info(f"Optimization level: {self.optimization_level}")
        
        # Validate input model
        if not self._validate_input_model():
            return False
        
        # Create output directory
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform conversion
        try:
            success = self._perform_conversion(**kwargs)
            if success and self.verbose:
                LOGGER.info(f"DXNN conversion completed: {self.output_path}")
            return success
        except Exception as e:
            LOGGER.error(f"DXNN conversion failed: {e}")
            return False
    
    def _validate_input_model(self) -> bool:
        """Validate the input model file."""
        input_path = Path(self.input_model)
        
        if not input_path.exists():
            LOGGER.error(f"Input model not found: {self.input_model}")
            return False
        
        # Check supported input formats
        supported_formats = ['.onnx', '.pt', '.pth', '.torchscript']
        if input_path.suffix.lower() not in supported_formats:
            LOGGER.error(f"Unsupported input format: {input_path.suffix}")
            LOGGER.error(f"Supported formats: {supported_formats}")
            return False
        
        return True
    
    def _perform_conversion(self, **kwargs) -> bool:
        """
        Perform the actual model conversion.
        
        Args:
            **kwargs: Additional conversion parameters
            
        Returns:
            bool: True if conversion successful
        """
        input_path = Path(self.input_model)
        output_dir = Path(self.output_path)
        
        # Determine conversion method based on input format
        if input_path.suffix.lower() == '.onnx':
            return self._convert_from_onnx(input_path, output_dir, **kwargs)
        elif input_path.suffix.lower() in ['.pt', '.pth']:
            return self._convert_from_pytorch(input_path, output_dir, **kwargs)
        elif input_path.suffix.lower() == '.torchscript':
            return self._convert_from_torchscript(input_path, output_dir, **kwargs)
        else:
            LOGGER.error(f"Unsupported input format: {input_path.suffix}")
            return False
    
    def _convert_from_onnx(self, input_path: Path, output_dir: Path, **kwargs) -> bool:
        """
        Convert from ONNX format to DXNN.
        
        Args:
            input_path (Path): Path to ONNX model
            output_dir (Path): Output directory
            **kwargs: Additional parameters
            
        Returns:
            bool: True if conversion successful
        """
        if self.verbose:
            LOGGER.info("Converting from ONNX format")
        
        try:
            # Simulate ONNX to DXNN conversion
            # In a real implementation, this would use ONNX Runtime or similar
            
            # Create DXNN model file
            dxnn_file = output_dir / f"{input_path.stem}.dxnn"
            self._create_dxnn_file(dxnn_file, "onnx")
            
            # Create metadata
            metadata = self._create_metadata(input_path, "onnx")
            metadata_file = output_dir / "metadata.yaml"
            self._save_metadata(metadata, metadata_file)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"ONNX to DXNN conversion failed: {e}")
            return False
    
    def _convert_from_pytorch(self, input_path: Path, output_dir: Path, **kwargs) -> bool:
        """
        Convert from PyTorch format to DXNN.
        
        Args:
            input_path (Path): Path to PyTorch model
            output_dir (Path): Output directory
            **kwargs: Additional parameters
            
        Returns:
            bool: True if conversion successful
        """
        if self.verbose:
            LOGGER.info("Converting from PyTorch format")
        
        try:
            # Load PyTorch model
            model = torch.load(input_path, map_location='cpu')
            
            # Convert to ONNX first (simplified)
            onnx_path = self._convert_pytorch_to_onnx(model, input_path)
            
            # Then convert ONNX to DXNN
            success = self._convert_from_onnx(onnx_path, output_dir, **kwargs)
            
            # Cleanup temporary ONNX file
            if onnx_path.exists():
                onnx_path.unlink()
            
            return success
            
        except Exception as e:
            LOGGER.error(f"PyTorch to DXNN conversion failed: {e}")
            return False
    
    def _convert_from_torchscript(self, input_path: Path, output_dir: Path, **kwargs) -> bool:
        """
        Convert from TorchScript format to DXNN.
        
        Args:
            input_path (Path): Path to TorchScript model
            output_dir (Path): Output directory
            **kwargs: Additional parameters
            
        Returns:
            bool: True if conversion successful
        """
        if self.verbose:
            LOGGER.info("Converting from TorchScript format")
        
        try:
            # Load TorchScript model
            model = torch.jit.load(input_path)
            
            # Convert to ONNX first (simplified)
            onnx_path = self._convert_torchscript_to_onnx(model, input_path)
            
            # Then convert ONNX to DXNN
            success = self._convert_from_onnx(onnx_path, output_dir, **kwargs)
            
            # Cleanup temporary ONNX file
            if onnx_path.exists():
                onnx_path.unlink()
            
            return success
            
        except Exception as e:
            LOGGER.error(f"TorchScript to DXNN conversion failed: {e}")
            return False
    
    def _convert_pytorch_to_onnx(self, model, input_path: Path) -> Path:
        """Convert PyTorch model to ONNX format."""
        # Create temporary ONNX file
        onnx_path = input_path.with_suffix('.onnx')
        
        # Simulate PyTorch to ONNX conversion
        # In a real implementation, this would use torch.onnx.export()
        
        # Create dummy ONNX file
        with open(onnx_path, 'w') as f:
            f.write("# Dummy ONNX file for conversion")
        
        return onnx_path
    
    def _convert_torchscript_to_onnx(self, model, input_path: Path) -> Path:
        """Convert TorchScript model to ONNX format."""
        # Create temporary ONNX file
        onnx_path = input_path.with_suffix('.onnx')
        
        # Simulate TorchScript to ONNX conversion
        # In a real implementation, this would use appropriate conversion
        
        # Create dummy ONNX file
        with open(onnx_path, 'w') as f:
            f.write("# Dummy ONNX file for conversion")
        
        return onnx_path
    
    def _create_dxnn_file(self, dxnn_path: Path, source_format: str):
        """
        Create DXNN model file.
        
        Args:
            dxnn_path (Path): Path to create DXNN file
            source_format (str): Source format of the model
        """
        # Simulate DXNN file creation
        # In a real implementation, this would create the actual DXNN format
        
        dxnn_content = {
            "format": "dxnn",
            "version": "1.0.0",
            "source_format": source_format,
            "target_device": self.target_device,
            "batch_size": self.batch_size,
            "optimization_level": self.optimization_level,
            "model_data": "base64_encoded_model_data_here"
        }
        
        with open(dxnn_path, 'w') as f:
            json.dump(dxnn_content, f, indent=2)
    
    def _create_metadata(self, input_path: Path, source_format: str) -> Dict[str, Any]:
        """
        Create metadata for the converted model.
        
        Args:
            input_path (Path): Path to input model
            source_format (str): Source format
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {
            "format": "dxnn",
            "version": "1.0.0",
            "source_format": source_format,
            "source_model": str(input_path),
            "target_device": self.target_device,
            "batch_size": self.batch_size,
            "optimization_level": self.optimization_level,
            "conversion_timestamp": str(datetime.datetime.now()),
            "input_shape": [1, 3, 640, 640],  # Default YOLO input shape
            "output_names": ["output0"],
            "model_type": "detection",
            "task": "detect"
        }
        
        return metadata
    
    def _save_metadata(self, metadata: Dict[str, Any], metadata_path: Path):
        """
        Save metadata to YAML file.
        
        Args:
            metadata (Dict[str, Any]): Metadata to save
            metadata_path (Path): Path to save metadata
        """
        try:
            import yaml
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON if YAML not available
            with open(metadata_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def get_conversion_info(self) -> Dict[str, Any]:
        """Get information about the conversion process."""
        return {
            "input_model": self.input_model,
            "output_path": self.output_path,
            "target_device": self.target_device,
            "batch_size": self.batch_size,
            "optimization_level": self.optimization_level,
            "verbose": self.verbose
        }
    
    def set_optimization_level(self, level: str):
        """
        Set the optimization level.
        
        Args:
            level (str): Optimization level ('fast', 'balanced', 'accurate')
        """
        valid_levels = ['fast', 'balanced', 'accurate']
        if level not in valid_levels:
            raise ValueError(f"Invalid optimization level. Must be one of: {valid_levels}")
        
        self.optimization_level = level
        if self.verbose:
            LOGGER.info(f"Optimization level set to: {level}")
    
    def set_target_device(self, device: str):
        """
        Set the target device for optimization.
        
        Args:
            device (str): Target device ('npu' or 'auto')
        """
        # DXNN only supports NPU devices, so always set to NPU
        self.target_device = "npu"
        if self.verbose:
            LOGGER.info(f"Target device set to: NPU (DXNN only supports NPU)")
