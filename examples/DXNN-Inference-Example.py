#!/usr/bin/env python3
"""
DXNN Inference Example for Ultralytics YOLO

This example demonstrates how to use the new DXNN model type for inference
in Ultralytics YOLO. DXNN is a hypothetical model format that supports
both CPU and GPU inference.

Requirements:
- ultralytics package
- dxnn-runtime package (hypothetical)
- A DXNN model file (*.dxnn or *_dxnn_model/ directory)

Usage:
    python DXNN-Inference-Example.py --model path/to/model.dxnn --source path/to/image.jpg
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="DXNN Inference Example")
    parser.add_argument("--model", type=str, required=True, help="Path to DXNN model file")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, cuda, or auto)")
    args = parser.parse_args()

    # Load the DXNN model
    print(f"Loading DXNN model from: {args.model}")
    model = YOLO(args.model)
    
    # Run inference
    print(f"Running inference on: {args.source}")
    # Note: For DXNN models, device handling is internal to the DXNN runtime
    # The device parameter here is for the main YOLO framework (cpu/cuda)
    results = model(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,  # Use standard PyTorch devices (cpu, cuda, auto)
        verbose=True
    )
    
    # Process results
    for result in results:
        if result.boxes is not None:
            print(f"Detected {len(result.boxes)} objects")
            
            # Get bounding boxes, confidence scores, and class labels
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Print detections
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = result.names[cls_id]
                print(f"  {i+1}. {class_name}: {conf:.3f} at {box}")
        else:
            print("No objects detected")
    
    print("DXNN inference completed successfully!")


if __name__ == "__main__":
    main()
