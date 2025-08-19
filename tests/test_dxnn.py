#!/usr/bin/env python3
"""
Test suite for DXNN module

This module provides comprehensive tests for the DXNN runtime, converter,
and utilities functionality.
"""

import json
import numpy as np
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from ultralytics.nn.dxnn import DXNNRuntime, DXNNConverter, DXNNUtils


class TestDXNNRuntime(unittest.TestCase):
    """Test cases for DXNNRuntime class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_model_path = Path(self.temp_dir) / "test_model.dxnn"
        self.create_test_model()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_model(self):
        """Create a test DXNN model file."""
        model_data = {
            "format": "dxnn",
            "version": "1.0.0",
            "source_format": "onnx",
            "target_device": "cpu",
            "batch_size": 1,
            "optimization_level": "balanced",
            "model_data": "base64_encoded_model_data_here"
        }
        
        with open(self.test_model_path, 'w') as f:
            json.dump(model_data, f)
    
    def test_init(self):
        """Test DXNNRuntime initialization."""
        runtime = DXNNRuntime(device="cpu", verbose=False)
        self.assertEqual(runtime.device, "npu")  # DXNN only supports NPU
        self.assertFalse(runtime._initialized)
        self.assertIsNone(runtime.model_path)
    
    def test_device_detection(self):
        """Test device detection logic."""
        runtime = DXNNRuntime(device="auto", verbose=False)
        
        # Test auto detection - should always be NPU
        self.assertEqual(runtime.device, "npu")
        
        # Test explicit device
        runtime = DXNNRuntime(device="npu", verbose=False)
        self.assertEqual(runtime.device, "npu")
        
        # Test invalid device - should default to NPU
        runtime = DXNNRuntime(device="cpu", verbose=False)
        self.assertEqual(runtime.device, "npu")  # DXNN only supports NPU
    
    def test_load_model_single_file(self):
        """Test loading model from single file."""
        runtime = DXNNRuntime(verbose=False)
        success = runtime.load_model(self.test_model_path)
        
        self.assertTrue(success)
        self.assertEqual(runtime.model_path, str(self.test_model_path))
        self.assertEqual(runtime.input_shape, (1, 3, 640, 640))
        self.assertEqual(runtime.output_names, ["output0"])
    
    def test_load_model_directory(self):
        """Test loading model from directory format."""
        # Create directory format
        model_dir = Path(self.temp_dir) / "test_model_dxnn_model"
        model_dir.mkdir()
        model_file = model_dir / "model.dxnn"
        self.create_test_model()
        shutil.copy(self.test_model_path, model_file)
        
        runtime = DXNNRuntime(verbose=False)
        success = runtime.load_model(model_dir)
        
        self.assertTrue(success)
        self.assertEqual(runtime.model_path, str(model_file))
    
    def test_initialize(self):
        """Test runtime initialization."""
        runtime = DXNNRuntime(verbose=False)
        runtime.load_model(self.test_model_path)
        success = runtime.initialize()
        
        self.assertTrue(success)
        self.assertTrue(runtime._initialized)
        self.assertIsNotNone(runtime._runtime)
    
    def test_inference(self):
        """Test model inference."""
        runtime = DXNNRuntime(verbose=False)
        runtime.load_model(self.test_model_path)
        runtime.initialize()
        
        # Test single input
        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = runtime.inference(input_data)
        
        self.assertIsInstance(outputs, list)
        self.assertGreater(len(outputs), 0)
        self.assertIsInstance(outputs[0], np.ndarray)
    
    def test_inference_batch(self):
        """Test batch inference."""
        runtime = DXNNRuntime(verbose=False)
        runtime.load_model(self.test_model_path)
        runtime.initialize()
        
        # Test batch input
        input_data = [np.random.randn(1, 3, 640, 640).astype(np.float32)]
        outputs = runtime.inference(input_data)
        
        self.assertIsInstance(outputs, list)
        self.assertGreater(len(outputs), 0)
    
    def test_invalid_input_shape(self):
        """Test inference with invalid input shape."""
        runtime = DXNNRuntime(verbose=False)
        runtime.load_model(self.test_model_path)
        runtime.initialize()
        
        # Test wrong input shape
        input_data = np.random.randn(1, 3, 320, 320).astype(np.float32)
        
        with self.assertRaises(ValueError):
            runtime.inference(input_data)
    
    def test_get_device_info(self):
        """Test device information retrieval."""
        runtime = DXNNRuntime(device="npu", verbose=False)
        runtime.load_model(self.test_model_path)
        runtime.initialize()
        
        device_info = runtime.get_device_info()
        
        self.assertIsInstance(device_info, dict)
        self.assertEqual(device_info["device"], "npu")
        self.assertTrue(device_info["initialized"])


class TestDXNNConverter(unittest.TestCase):
    """Test cases for DXNNConverter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_onnx_path = Path(self.temp_dir) / "test_model.onnx"
        self.test_pt_path = Path(self.temp_dir) / "test_model.pt"
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test model files."""
        # Create dummy ONNX file
        with open(self.test_onnx_path, 'w') as f:
            f.write("# Dummy ONNX file")
        
        # Create dummy PyTorch file
        dummy_model = torch.nn.Linear(10, 1)
        torch.save(dummy_model.state_dict(), self.test_pt_path)
    
    def test_init(self):
        """Test DXNNConverter initialization."""
        converter = DXNNConverter(verbose=False)
        self.assertIsNone(converter.input_model)
        self.assertIsNone(converter.output_path)
        self.assertEqual(converter.target_device, "npu")  # DXNN only supports NPU
        self.assertEqual(converter.batch_size, 1)
    
    def test_validate_input_model(self):
        """Test input model validation."""
        converter = DXNNConverter(verbose=False)
        
        # Test valid ONNX file
        converter.input_model = str(self.test_onnx_path)
        self.assertTrue(converter._validate_input_model())
        
        # Test non-existent file
        converter.input_model = "non_existent.onnx"
        self.assertFalse(converter._validate_input_model())
        
        # Test unsupported format
        unsupported_path = Path(self.temp_dir) / "test.txt"
        unsupported_path.touch()
        converter.input_model = str(unsupported_path)
        self.assertFalse(converter._validate_input_model())
    
    def test_convert_from_onnx(self):
        """Test ONNX to DXNN conversion."""
        converter = DXNNConverter(verbose=False)
        output_path = Path(self.temp_dir) / "output_dxnn_model"
        
        success = converter.convert(
            input_model=self.test_onnx_path,
            output_path=output_path,
            target_device="npu",  # DXNN only supports NPU
            batch_size=1
        )
        
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
        
        # Check for DXNN file
        dxnn_files = list(output_path.glob("*.dxnn"))
        self.assertGreater(len(dxnn_files), 0)
        
        # Check for metadata
        metadata_files = list(output_path.glob("metadata.*"))
        self.assertGreater(len(metadata_files), 0)
    
    def test_convert_from_pytorch(self):
        """Test PyTorch to DXNN conversion."""
        converter = DXNNConverter(verbose=False)
        output_path = Path(self.temp_dir) / "output_dxnn_model"
        
        success = converter.convert(
            input_model=self.test_pt_path,
            output_path=output_path,
            target_device="npu",  # DXNN only supports NPU
            batch_size=1
        )
        
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
    
    def test_set_optimization_level(self):
        """Test optimization level setting."""
        converter = DXNNConverter(verbose=False)
        
        # Test valid levels
        for level in ["fast", "balanced", "accurate"]:
            converter.set_optimization_level(level)
            self.assertEqual(converter.optimization_level, level)
        
        # Test invalid level
        with self.assertRaises(ValueError):
            converter.set_optimization_level("invalid")
    
    def test_set_target_device(self):
        """Test target device setting."""
        converter = DXNNConverter(verbose=False)
        
        # Test valid devices
        for device in ["npu", "auto"]:
            converter.set_target_device(device)
            self.assertEqual(converter.target_device, "npu")  # Always NPU for DXNN
        
        # Test invalid device - should default to NPU
        converter.set_target_device("cpu")
        self.assertEqual(converter.target_device, "npu")


class TestDXNNUtils(unittest.TestCase):
    """Test cases for DXNNUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_model_path = Path(self.temp_dir) / "test_model.dxnn"
        self.create_test_model()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_model(self):
        """Create a test DXNN model file."""
        model_data = {
            "format": "dxnn",
            "version": "1.0.0",
            "model_data": "base64_encoded_model_data_here"
        }
        
        with open(self.test_model_path, 'w') as f:
            json.dump(model_data, f)
    
    def test_validate_dxnn_model(self):
        """Test DXNN model validation."""
        # Test valid model
        result = DXNNUtils.validate_dxnn_model(self.test_model_path)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        
        # Test non-existent model
        result = DXNNUtils.validate_dxnn_model("non_existent.dxnn")
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)
        
        # Test invalid JSON
        invalid_path = Path(self.temp_dir) / "invalid.dxnn"
        with open(invalid_path, 'w') as f:
            f.write("invalid json content")
        
        result = DXNNUtils.validate_dxnn_model(invalid_path)
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)
    
    def test_get_system_info(self):
        """Test system information retrieval."""
        info = DXNNUtils.get_system_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("platform", info)
        self.assertIn("architecture", info)
        self.assertIn("python_version", info)
        self.assertIn("torch_version", info)
        self.assertIn("cuda_available", info)
        self.assertIn("cpu_count", info)
    
    def test_profile_inference_time(self):
        """Test inference time profiling."""
        runtime = DXNNRuntime(verbose=False)
        runtime.load_model(self.test_model_path)
        runtime.initialize()
        
        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        timing_stats = DXNNUtils.profile_inference_time(runtime, input_data, num_runs=5)
        
        self.assertIsInstance(timing_stats, dict)
        self.assertIn("mean", timing_stats)
        self.assertIn("std", timing_stats)
        self.assertIn("min", timing_stats)
        self.assertIn("max", timing_stats)
        self.assertIn("median", timing_stats)
    
    def test_benchmark_model(self):
        """Test model benchmarking."""
        benchmark_result = DXNNUtils.benchmark_model(
            self.test_model_path,
            input_shape=(1, 3, 640, 640),
            num_runs=5,
            device="npu"
        )
        
        self.assertIsInstance(benchmark_result, dict)
        self.assertIn("model_path", benchmark_result)
        self.assertIn("timing_stats", benchmark_result)
        self.assertIn("system_info", benchmark_result)
        self.assertIn("success", benchmark_result)
    
    def test_compare_models(self):
        """Test model comparison."""
        # Create second test model
        test_model_path2 = Path(self.temp_dir) / "test_model2.dxnn"
        self.create_test_model()
        import shutil
        shutil.copy(self.test_model_path, test_model_path2)
        
        comparison_result = DXNNUtils.compare_models(
            [self.test_model_path, test_model_path2],
            input_shape=(1, 3, 640, 640),
            num_runs=5
        )
        
        self.assertIsInstance(comparison_result, dict)
        self.assertIn("models", comparison_result)
        self.assertIn("summary", comparison_result)
        self.assertIn("success", comparison_result)
    
    def test_optimize_model_config(self):
        """Test model configuration optimization."""
        optimization_result = DXNNUtils.optimize_model_config(
            self.test_model_path,
            target_device="npu",
            optimization_goals=["speed", "memory"]
        )
        
        self.assertIsInstance(optimization_result, dict)
        self.assertIn("model_path", optimization_result)
        self.assertIn("recommendations", optimization_result)
        self.assertIn("success", optimization_result)
    
    def test_export_benchmark_report(self):
        """Test benchmark report export."""
        benchmark_results = {
            "model_path": str(self.test_model_path),
            "device": "npu",
            "timing_stats": {
                "mean": 0.1,
                "std": 0.01,
                "min": 0.09,
                "max": 0.11
            },
            "system_info": {
                "platform": "test",
                "cpu_count": 4
            }
        }
        
        report_path = Path(self.temp_dir) / "benchmark_report.md"
        success = DXNNUtils.export_benchmark_report(benchmark_results, report_path)
        
        self.assertTrue(success)
        self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
