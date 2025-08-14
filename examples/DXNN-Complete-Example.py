#!/usr/bin/env python3
"""
Complete DXNN Module Example

This example demonstrates the full capabilities of the DXNN module including
model conversion, inference, benchmarking, and optimization.
"""

import argparse
import json
import numpy as np
import tempfile
import time
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.dxnn import DXNNRuntime, DXNNConverter, DXNNUtils


def create_dummy_model():
    """Create a dummy PyTorch model for testing."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 84, 1),  # YOLO-like output
        torch.nn.AdaptiveAvgPool2d((1, 8400))
    )
    return model


def create_test_model_file(model_path: Path):
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
    
    with open(model_path, 'w') as f:
        json.dump(model_data, f)


def demonstrate_runtime():
    """Demonstrate DXNN runtime functionality."""
    print("\n" + "="*50)
    print("DXNN Runtime Demonstration")
    print("="*50)
    
    # Create temporary test model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.dxnn"
        create_test_model_file(model_path)
        
        # Initialize runtime
        print("1. Initializing DXNN Runtime...")
        runtime = DXNNRuntime(device="npu", verbose=True)
        
        # Load model
        print("2. Loading DXNN model...")
        success = runtime.load_model(model_path)
        print(f"   Model loaded: {success}")
        
        # Initialize runtime
        print("3. Initializing runtime...")
        success = runtime.initialize()
        print(f"   Runtime initialized: {success}")
        
        # Get device info
        print("4. Device information:")
        device_info = runtime.get_device_info()
        for key, value in device_info.items():
            print(f"   {key}: {value}")
        
        # Run inference
        print("5. Running inference...")
        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        start_time = time.time()
        outputs = runtime.inference(input_data)
        inference_time = time.time() - start_time
        
        print(f"   Inference completed in {inference_time:.4f}s")
        print(f"   Number of outputs: {len(outputs)}")
        print(f"   Output shapes: {[out.shape for out in outputs]}")


def demonstrate_converter():
    """Demonstrate DXNN converter functionality."""
    print("\n" + "="*50)
    print("DXNN Converter Demonstration")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dummy PyTorch model
        print("1. Creating dummy PyTorch model...")
        model = create_dummy_model()
        pt_path = temp_path / "dummy_model.pt"
        torch.save(model.state_dict(), pt_path)
        
        # Create dummy ONNX file
        print("2. Creating dummy ONNX file...")
        onnx_path = temp_path / "dummy_model.onnx"
        with open(onnx_path, 'w') as f:
            f.write("# Dummy ONNX file for conversion")
        
        # Initialize converter
        print("3. Initializing DXNN Converter...")
        converter = DXNNConverter(verbose=True)
        
        # Convert from PyTorch
        print("4. Converting PyTorch model to DXNN...")
        output_path = temp_path / "converted_from_pytorch"
        success = converter.convert(
            input_model=pt_path,
            output_path=output_path,
            target_device="npu",
            batch_size=1,
            optimization_level="balanced"
        )
        print(f"   PyTorch conversion: {'Success' if success else 'Failed'}")
        
        # Convert from ONNX
        print("5. Converting ONNX model to DXNN...")
        output_path = temp_path / "converted_from_onnx"
        success = converter.convert(
            input_model=onnx_path,
            output_path=output_path,
            target_device="npu",
            batch_size=1,
            optimization_level="balanced"
        )
        print(f"   ONNX conversion: {'Success' if success else 'Failed'}")
        
        # Show conversion info
        print("6. Conversion information:")
        conversion_info = converter.get_conversion_info()
        for key, value in conversion_info.items():
            print(f"   {key}: {value}")


def demonstrate_utils():
    """Demonstrate DXNN utilities functionality."""
    print("\n" + "="*50)
    print("DXNN Utilities Demonstration")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model
        model_path = temp_path / "test_model.dxnn"
        create_test_model_file(model_path)
        
        # Model validation
        print("1. Validating DXNN model...")
        validation_result = DXNNUtils.validate_dxnn_model(model_path)
        print(f"   Valid: {validation_result['valid']}")
        if validation_result['errors']:
            print(f"   Errors: {validation_result['errors']}")
        if validation_result['warnings']:
            print(f"   Warnings: {validation_result['warnings']}")
        
        # System information
        print("2. System information:")
        system_info = DXNNUtils.get_system_info()
        for key, value in system_info.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")
        
        # Model benchmarking
        print("3. Benchmarking model...")
        benchmark_result = DXNNUtils.benchmark_model(
            model_path,
            input_shape=(1, 3, 640, 640),
            num_runs=10,
            device="npu"
        )
        
        if benchmark_result['success']:
            print("   Benchmark results:")
            timing_stats = benchmark_result['timing_stats']
            for key, value in timing_stats.items():
                print(f"     {key}: {value:.4f}s")
        else:
            print(f"   Benchmark failed: {benchmark_result.get('error', 'Unknown error')}")
        
        # Model optimization
        print("4. Optimizing model configuration...")
        optimization_result = DXNNUtils.optimize_model_config(
            model_path,
            target_device="npu",
            optimization_goals=["speed", "memory"]
        )
        
        if optimization_result['success']:
            print("   Optimization recommendations:")
            recommendations = optimization_result['recommendations']
            for key, value in recommendations.items():
                print(f"     {key}: {value}")
        else:
            print(f"   Optimization failed: {optimization_result.get('error', 'Unknown error')}")


def demonstrate_integration():
    """Demonstrate DXNN integration with Ultralytics."""
    print("\n" + "="*50)
    print("DXNN Ultralytics Integration Demonstration")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a DXNN model directory
        model_dir = temp_path / "yolo11n_dxnn_model"
        model_dir.mkdir()
        
        # Create DXNN model file
        dxnn_file = model_dir / "yolo11n.dxnn"
        create_test_model_file(dxnn_file)
        
        # Create metadata
        metadata = {
            "format": "dxnn",
            "version": "1.0.0",
            "task": "detect",
            "stride": 32,
            "batch": 1,
            "imgsz": [640, 640],
            "names": {i: f"class{i}" for i in range(80)},
            "model_type": "detection"
        }
        
        metadata_file = model_dir / "metadata.yaml"
        import yaml
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        
        print("1. Created DXNN model directory structure:")
        print(f"   Model directory: {model_dir}")
        print(f"   DXNN file: {dxnn_file}")
        print(f"   Metadata file: {metadata_file}")
        
        # Demonstrate loading with Ultralytics
        print("2. Loading DXNN model with Ultralytics YOLO...")
        try:
            model = YOLO(str(model_dir))
            print("   Model loaded successfully!")
            
            # Run inference
            print("3. Running inference with Ultralytics...")
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(dummy_image, verbose=False)
            print(f"   Inference completed! Number of results: {len(results)}")
            
        except Exception as e:
            print(f"   Error loading model: {e}")
            print("   Note: This is expected since we're using a dummy model")


def demonstrate_performance_comparison():
    """Demonstrate performance comparison between models."""
    print("\n" + "="*50)
    print("DXNN Performance Comparison Demonstration")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple test models
        model_paths = []
        for i in range(3):
            model_path = temp_path / f"test_model_{i}.dxnn"
            create_test_model_file(model_path)
            model_paths.append(model_path)
        
        print(f"1. Created {len(model_paths)} test models for comparison")
        
        # Compare models
        print("2. Comparing model performance...")
        comparison_result = DXNNUtils.compare_models(
            model_paths,
            input_shape=(1, 3, 640, 640),
            num_runs=5
        )
        
        if comparison_result['success']:
            print("   Comparison results:")
            summary = comparison_result['summary']
            print(f"     Fastest model: {summary['fastest_model']}")
            print(f"     Slowest model: {summary['slowest_model']}")
            print(f"     Speedup factor: {summary['speedup_factor']:.2f}x")
        else:
            print(f"   Comparison failed: {comparison_result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description="DXNN Module Complete Example")
    parser.add_argument("--demo", choices=["runtime", "converter", "utils", "integration", "performance", "all"],
                       default="all", help="Which demonstration to run")
    args = parser.parse_args()
    
    print("DXNN Module Complete Example")
    print("This example demonstrates all features of the DXNN module")
    
    if args.demo == "all" or args.demo == "runtime":
        demonstrate_runtime()
    
    if args.demo == "all" or args.demo == "converter":
        demonstrate_converter()
    
    if args.demo == "all" or args.demo == "utils":
        demonstrate_utils()
    
    if args.demo == "all" or args.demo == "integration":
        demonstrate_integration()
    
    if args.demo == "all" or args.demo == "performance":
        demonstrate_performance_comparison()
    
    print("\n" + "="*50)
    print("DXNN Module Example Completed!")
    print("="*50)
    print("\nKey Features Demonstrated:")
    print("✓ DXNN Runtime for model inference")
    print("✓ DXNN Converter for model conversion")
    print("✓ DXNN Utilities for validation and benchmarking")
    print("✓ Integration with Ultralytics YOLO")
    print("✓ Performance comparison and optimization")
    print("\nThe DXNN module provides a complete solution for")
    print("efficient model inference across various platforms.")


if __name__ == "__main__":
    main()
