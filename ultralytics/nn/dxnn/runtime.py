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

from dx_engine import InferenceEngine
import cv2
import json


PPU_TYPES = ["BBOX", "POSE"]

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
        LOGGER.setLevel(logging.DEBUG)

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

        # --- Load JSON config with the same base name as the model file ---
        json_config_path = model_path.with_suffix('.json')
        if json_config_path.exists():
            with open(json_config_path, 'r') as f:
                self.config = json.load(f)
            if self.verbose:
                LOGGER.info(f"Loaded config from {json_config_path}")
        else:
            self.config = None
            if self.verbose:
                LOGGER.info(f"No config file found at {json_config_path}")

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
        """Create and return the NPU runtime using InferenceEngine."""
        
        if not self.model_path:
            raise RuntimeError("Model path not set. Call load_model() first.")
        try:
            ie = InferenceEngine(self.model_path)
            self.input_size = int(np.sqrt(ie.get_input_size() / 3))

            self.pp_type = self.config["model"]["param"]["decoding_method"]
            self.layers = self.config["model"]["param"]["layer"]
            self.classes = self.config["output"]["classes"]
            self.n_classes = len(self.classes)

            if self.verbose:
                print(f"Model loaded successfully: {self.model_path}")
                print(f"Input size: {self.input_size}")
            return ie
        except Exception as e:
            print(f"Failed to initialize InferenceEngine: {e}")
            raise
    
    def letter_box(self, image_src, new_shape=(512, 512), fill_color=(114, 114, 114), format=None):
        src_shape = image_src.shape[1:3] # height, width
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])

        ratio = r, r
        new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = new_shape[1], new_shape[0]

        dw /= 2
        dh /= 2

        if src_shape[::-1] != new_unpad:
            image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image_new = cv2.copyMakeBorder(image_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)  # add border
        if format is not None:
            image_new = cv2.cvtColor(image_new, format)

        return image_new, ratio, (dw, dh)


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
                    inputs[i] = input_tensor # = np.transpose(input_tensor, (0, 3, 1, 2))
                else:
                    # If shapes don't match, try to adapt the input shape to match the actual input
                    if self.verbose:
                        LOGGER.warning(f"Input shape {input_tensor.shape[1:]} doesn't match expected {self.input_shape[1:]}, adapting...")
                    
                    # Convert HWC to CHW if needed
                    if input_tensor.shape[1] == 3:  # HWC format
                        inputs[i] = np.transpose(input_tensor, (0, 3, 1, 2))
                    # Update input shape to match actual input
                    self.input_shape = input_tensor.shape
        
        # ie_inputs = [self.letter_box(input_tensor, new_shape=(self.input_size, self.input_size), fill_color=(114, 114, 114), format=None)[0] for input_tensor in inputs]
        ie_inputs = inputs
        ie_outputs = self._runtime.run(ie_inputs)
        
        if self.verbose:
            LOGGER.info(f"DXNN_IE inference completed: {len(ie_outputs)} outputs")
        
        decoded_outputs = []
        ppu_type = (self._runtime.get_output_tensors_info()[0]['dtype'])
        # ppu_type = "BBOX"

        if ppu_type in PPU_TYPES:
            if self.pp_type in ["yolo_pose"]:
                decoded_outputs = DXNNRuntime.ppu_decode_pose(ie_outputs, self.layers, self.n_classes)
            else:
                decoded_outputs = DXNNRuntime.ppu_decode(ie_outputs, self.layers, self.n_classes)
        elif len(ie_outputs) > 1:
            cpu_model_path = os.path.join(os.path.split(self.model_path)[0], "cpu_0.onnx")
            if os.path.exists(cpu_model_path):
                decoded_outputs = DXNNRuntime.onnx_decode(ie_outputs, cpu_model_path)
            else:
                decoded_outputs = DXNNRuntime.all_decode(ie_outputs, self.layers, self.n_classes)
        elif len(ie_outputs) == 1:
            decoded_outputs = ie_outputs[0]
        else:
            raise ValueError(f"[Error] Output Size {len(ie_outputs )} is not supported !!")
        print("decoding output Done! ")

        if self.verbose:
            LOGGER.info(f"DXNN inference completed: {len(decoded_outputs)} outputs")
        
        outputs = decoded_outputs.T
        outputs = outputs[np.newaxis, ...]
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


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def ppu_decode_pose(ie_outputs, layer_config, n_classes):
        ie_output = ie_outputs[0][0]
        num_det = ie_output.shape[0]
        decoded_tensor = []
        for detected_idx in range(num_det):
            tensor = np.zeros((n_classes + 5), dtype=float)

            data = ie_output[detected_idx].tobytes()

            box = np.frombuffer(data[0:16], np.float32)
            gy, gx, anchor, layer = np.frombuffer(data[16:20], np.uint8)
            score = np.frombuffer(data[20:24], np.float32)
            label = 0 # np.frombuffer(data[24:28], np.uint32)
            kpts = np.frombuffer(data[28:232], np.float32)

            if layer > len(layer_config):
                break

            w = layer_config[layer]["anchor_width"][anchor]
            h = layer_config[layer]["anchor_height"][anchor]
            s = layer_config[layer]["stride"]

            grid = np.array([gx, gy], np.float32)
            anchor_wh = np.array([w, h], np.float32)
            xc = (grid - 0.5 + (box[0:2] * 2)) * s
            wh = box[2:4] ** 2 * 4 * anchor_wh

            box = np.concatenate([xc, wh], axis=0)
            tensor[:4] = box
            tensor[4] = score
            tensor[4+1+label] = score

            # for i in range(17):
            #     start = (n_classes + 5) + (i * 3)
            #     tensor[start:start+2] = (grid - 0.5 + (kpts[i*3:i*3+2] * 2)) * s
            #     tensor[start+2:start+3] = kpts[i*3+2:i*3+3]

            decoded_tensor.append(tensor)
        if len(decoded_tensor) == 0:
            decoded_tensor = np.zeros((n_classes + 5), dtype=float)

        decoded_output = np.stack(decoded_tensor)

        return decoded_output

    @staticmethod
    def ppu_decode(ie_outputs, layer_config, n_classes):
        ie_output = ie_outputs[0][0]
        num_det = ie_output.shape[0]
        decoded_tensor = []
        for detected_idx in range(num_det):
            tensor = np.zeros((n_classes + 5), dtype=float)
            data = ie_output[detected_idx].tobytes()
            box = np.frombuffer(data[0:16], np.float32)
            gy, gx, anchor, layer = np.frombuffer(data[16:20], np.uint8)
            score = np.frombuffer(data[20:24], np.float32)
            label = np.frombuffer(data[24:28], np.uint32)
            if layer > len(layer_config):
                break
            w = layer_config[layer]["anchor_width"][anchor]
            h = layer_config[layer]["anchor_height"][anchor]
            s = layer_config[layer]["stride"]

            grid = np.array([gx, gy], np.float32)
            anchor_wh = np.array([w, h], np.float32)
            xc = (grid - 0.5 + (box[0:2] * 2)) * s
            wh = box[2:4] ** 2 * 4 * anchor_wh
            box = np.concatenate([xc, wh], axis=0)
            tensor[:4] = box
            tensor[4] = score
            tensor[4+1+label] = score
            decoded_tensor.append(tensor)
        if len(decoded_tensor) == 0:
            decoded_tensor = np.zeros((n_classes + 5), dtype=float)

        decoded_output = np.stack(decoded_tensor)

        return decoded_output

    @staticmethod
    def all_decode(ie_outputs, layer_config, n_classes):
        ''' slice outputs'''
        outputs = []
        outputs.append(ie_outputs[0][...,:(n_classes + 5)* len(layer_config[0]["anchor_width"])])
        outputs.append(ie_outputs[1][...,:(n_classes + 5)* len(layer_config[0]["anchor_width"])])
        outputs.append(ie_outputs[2][...,:(n_classes + 5)* len(layer_config[0]["anchor_width"])])

        decoded_tensor = []

        for i, output in enumerate(outputs):
            for l in range(len(layer_config[i]["anchor_width"])):
                start = l*(n_classes + 5)
                end = start + n_classes + 5

                layer = layer_config[i]
                stride = layer["stride"]
                grid_size = output.shape[2]
                meshgrid_x = np.arange(0, grid_size)
                meshgrid_y = np.arange(0, grid_size)
                grid = np.stack([np.meshgrid(meshgrid_y, meshgrid_x)], axis=-1)[...,0]
                output[...,start+4:end] = DXNNRuntime.sigmoid(output[...,start+4:end])
                cxcy = output[...,start+0:start+2]
                wh = output[...,start+2:start+4]
                cxcy[...,0] = (DXNNRuntime.sigmoid(cxcy[...,0]) * 2 - 0.5 + grid[0]) * stride
                cxcy[...,1] = (DXNNRuntime.sigmoid(cxcy[...,1]) * 2 - 0.5 + grid[1]) * stride
                wh[...,0] = ((DXNNRuntime.sigmoid(wh[...,0]) * 2) ** 2) * layer["anchor_width"][l]
                wh[...,1] = ((DXNNRuntime.sigmoid(wh[...,1]) * 2) ** 2) * layer["anchor_height"][l]
                decoded_tensor.append(output[...,start+0:end].reshape(-1, n_classes + 5))

        decoded_output = np.concatenate(decoded_tensor, axis=0)

        return decoded_output

    @staticmethod
    def onnx_decode(ie_outputs, cpu_onnx_path):
        import onnxruntime as ort
        sess = ort.InferenceSession(cpu_onnx_path)
        input_names = [input.name for input in sess.get_inputs()]
        input_dict = {input_names[0]:ie_outputs[0], input_names[1]:ie_outputs[1], input_names[2]:ie_outputs[2]}
        ort_output = sess.run(None, input_dict)
        return ort_output[0][0]

    
    def __del__(self):
        """Cleanup when the runtime is destroyed."""
        if self._initialized and self.verbose:
            LOGGER.debug("DXNN runtime destroyed")
