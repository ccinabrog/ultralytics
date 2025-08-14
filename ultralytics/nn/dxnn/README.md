# DXNN Module for Ultralytics

The DXNN (Deep Neural Network eXecution) module provides a complete solution for efficient model inference across various hardware platforms. This module is designed to work seamlessly with Ultralytics YOLO models and provides both runtime inference capabilities and model conversion utilities.

## Features

- **NPU inference**: Optimized for Neural Processing Units
- **Model conversion**: Convert from ONNX, PyTorch, and TorchScript formats
- **Performance optimization**: Automatic device detection and optimization
- **Benchmarking tools**: Comprehensive performance profiling and comparison
- **Easy integration**: Seamless integration with Ultralytics YOLO pipeline

## Installation

The DXNN module is included with Ultralytics and will be automatically installed when needed:

```bash
pip install ultralytics
```

## Quick Start

### Basic Inference

```python
from ultralytics.nn.dxnn import DXNNRuntime

# Initialize runtime
runtime = DXNNRuntime(device="npu")

# Load model
runtime.load_model("model.dxnn")
runtime.initialize()

# Run inference
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = runtime.inference(input_data)
```

### Model Conversion

```python
from ultralytics.nn.dxnn import DXNNConverter

# Initialize converter
converter = DXNNConverter()

# Convert from ONNX
converter.convert(
    input_model="model.onnx",
    output_path="output_dxnn_model",
    target_device="npu",
    batch_size=1
)
```

### Integration with Ultralytics

```python
from ultralytics import YOLO

# Load DXNN model directly
model = YOLO("model_dxnn_model/")

# Run inference
results = model("image.jpg")
```

## Module Components

### 1. DXNNRuntime

The core inference engine for DXNN models.

**Key Features:**
- NPU-optimized inference
- Batch processing support
- Memory optimization
- Error handling and validation

**Usage:**
```python
from ultralytics.nn.dxnn import DXNNRuntime

runtime = DXNNRuntime(device="npu", verbose=True)
runtime.load_model("model.dxnn")
runtime.initialize()

# Single inference
outputs = runtime.inference(input_data)

# Batch inference
batch_outputs = runtime.inference([input1, input2, input3])
```

### 2. DXNNConverter

Model conversion utility for creating DXNN models from various formats.

**Supported Input Formats:**
- ONNX (.onnx)
- PyTorch (.pt, .pth)
- TorchScript (.torchscript)

**Usage:**
```python
from ultralytics.nn.dxnn import DXNNConverter

converter = DXNNConverter(verbose=True)

# Convert with custom settings
converter.convert(
    input_model="model.onnx",
    output_path="output_dxnn_model",
    target_device="npu",
    batch_size=4,
    optimization_level="fast"
)
```

### 3. DXNNUtils

Utility functions for model validation, benchmarking, and optimization.

**Key Functions:**
- Model validation
- Performance benchmarking
- System information
- Configuration optimization

**Usage:**
```python
from ultralytics.nn.dxnn import DXNNUtils

# Validate model
validation = DXNNUtils.validate_dxnn_model("model.dxnn")
print(f"Model valid: {validation['valid']}")

# Benchmark performance
benchmark = DXNNUtils.benchmark_model(
    "model.dxnn",
    input_shape=(1, 3, 640, 640),
    num_runs=100,
    device="npu"
)

# Compare models
comparison = DXNNUtils.compare_models([
    "model1.dxnn",
    "model2.dxnn",
    "model3.dxnn"
])
```

## Model Formats

### Single File Format
```
model.dxnn
```

### Directory Format
```
model_dxnn_model/
├── model.dxnn
└── metadata.yaml
```

## Performance Optimization

### Device Selection

DXNN is optimized for NPU devices:

```python
# Use NPU device
runtime = DXNNRuntime(device="npu")

# Auto-detect (defaults to NPU)
runtime = DXNNRuntime(device="auto")
```

### Optimization Levels

```python
converter = DXNNConverter()

# Fast optimization (speed over accuracy)
converter.set_optimization_level("fast")

# Balanced optimization (default)
converter.set_optimization_level("balanced")

# Accurate optimization (accuracy over speed)
converter.set_optimization_level("accurate")
```

## Examples

### Complete Example

See `examples/DXNN-Complete-Example.py` for a comprehensive demonstration of all features.

### Basic Inference Example

```python
import numpy as np
from ultralytics.nn.dxnn import DXNNRuntime

# Initialize and load model
runtime = DXNNRuntime(device="npu")
runtime.load_model("model.dxnn")
runtime.initialize()

# Prepare input data
input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Run inference
outputs = runtime.inference(input_data)

# Process results
for i, output in enumerate(outputs):
    print(f"Output {i}: shape {output.shape}")
```

### Model Conversion Example

```python
from ultralytics.nn.dxnn import DXNNConverter

# Convert ONNX model to DXNN
converter = DXNNConverter()
success = converter.convert(
    input_model="yolo11n.onnx",
    output_path="yolo11n_dxnn_model",
    target_device="npu",
    batch_size=1,
    optimization_level="balanced"
)

if success:
    print("Conversion completed successfully!")
```

### Benchmarking Example

```python
from ultralytics.nn.dxnn import DXNNUtils

# Benchmark single model
benchmark = DXNNUtils.benchmark_model(
    "model.dxnn",
    input_shape=(1, 3, 640, 640),
    num_runs=100,
    device="npu"
)

print(f"Average inference time: {benchmark['timing_stats']['mean']:.4f}s")

# Compare multiple models
comparison = DXNNUtils.compare_models([
    "model1.dxnn",
    "model2.dxnn"
])

print(f"Fastest model: {comparison['summary']['fastest_model']}")
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_dxnn.py -v
```

Or run individual test classes:

```bash
python -m pytest tests/test_dxnn.py::TestDXNNRuntime -v
python -m pytest tests/test_dxnn.py::TestDXNNConverter -v
python -m pytest tests/test_dxnn.py::TestDXNNUtils -v
```

## Integration with Ultralytics

The DXNN module is fully integrated with Ultralytics and can be used seamlessly:

### Export to DXNN

```python
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Export to DXNN
model.export(format="dxnn", device="npu")
```

### Load DXNN Model

```python
from ultralytics import YOLO

# Load DXNN model
model = YOLO("yolo11n_dxnn_model/")

# Run inference
results = model("image.jpg")
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the DXNN model file exists and is accessible
2. **NPU not available**: Check if NPU device is available and properly configured
3. **Memory issues**: Reduce batch size or image resolution
4. **Conversion failures**: Verify input model format is supported

### Performance Tips

1. **Optimize for NPU**: DXNN is specifically designed for NPU acceleration
2. **Optimize batch size**: Balance between memory usage and throughput
3. **Choose optimization level**: Fast for speed, Accurate for precision
4. **Profile performance**: Use benchmarking tools to identify bottlenecks

## API Reference

### DXNNRuntime

#### Methods

- `__init__(device="auto", verbose=True)`: Initialize runtime
- `load_model(model_path)`: Load DXNN model
- `initialize()`: Initialize runtime for inference
- `inference(inputs)`: Run inference on input data
- `get_device_info()`: Get device information
- `set_batch_size(batch_size)`: Set batch size

#### Attributes

- `model_path`: Path to loaded model
- `device`: Current inference device
- `input_shape`: Expected input shape
- `output_names`: Output tensor names
- `metadata`: Model metadata

### DXNNConverter

#### Methods

- `__init__(verbose=True)`: Initialize converter
- `convert(input_model, output_path, **kwargs)`: Convert model
- `set_optimization_level(level)`: Set optimization level
- `set_target_device(device)`: Set target device
- `get_conversion_info()`: Get conversion information

#### Parameters

- `input_model`: Path to input model
- `output_path`: Path for output DXNN model
- `target_device`: Target device for optimization (npu)
- `batch_size`: Batch size for converted model
- `optimization_level`: Optimization level (fast/balanced/accurate)

### DXNNUtils

#### Static Methods

- `validate_dxnn_model(model_path)`: Validate DXNN model
- `get_system_info()`: Get system information
- `profile_inference_time(runtime, input_data, **kwargs)`: Profile inference time
- `benchmark_model(model_path, **kwargs)`: Benchmark model performance
- `compare_models(model_paths, **kwargs)`: Compare model performance
- `optimize_model_config(model_path, **kwargs)`: Optimize model configuration
- `export_benchmark_report(benchmark_results, output_path)`: Export benchmark report

## Contributing

To contribute to the DXNN module:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This module is part of Ultralytics and is licensed under the AGPL-3.0 License.

## Support

For support and questions:

- Check the [Ultralytics documentation](https://docs.ultralytics.com/)
- Visit the [Ultralytics community](https://community.ultralytics.com/)
- Open an issue on [GitHub](https://github.com/ultralytics/ultralytics)
