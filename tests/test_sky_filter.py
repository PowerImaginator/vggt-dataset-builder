"""
Test script for sky filtering
Usage: python test_sky_filter.py <image_path>
"""
import sys
import numpy as np
import cv2
from pathlib import Path

def test_sky_filter(image_path):
    print(f"Testing sky filter on: {image_path}")
    
    # Check if onnxruntime is available
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Install with: pip install onnxruntime")
        return
    
    # Load the model
    skyseg_model_path = Path(__file__).parent / "skyseg.onnx"
    if not skyseg_model_path.exists():
        print(f"Downloading sky segmentation model to {skyseg_model_path}...")
        import urllib.request
        model_url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
        urllib.request.urlretrieve(model_url, skyseg_model_path)
        print(f"Downloaded sky segmentation model")
    
    print(f"Loading model from {skyseg_model_path}")
    skyseg_session = ort.InferenceSession(str(skyseg_model_path), providers=['CPUExecutionProvider'])
    
    # Load image
    print(f"Loading image...")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image from {image_path}")
        return
    
    print(f"Image shape: {img.shape} (H={img.shape[0]}, W={img.shape[1]})")
    
    # Resize to 320x320 for sky segmentation model
    print("Resizing to 320x320...")
    img_resized = cv2.resize(img, (320, 320))
    
    # Convert BGR to RGB and normalize with PyTorch stats
    print("Converting and normalizing...")
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_input = (img_input / 255.0 - mean) / std
    
    # Transpose to CHW and add batch dimension
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    print(f"Input tensor shape: {img_input.shape}")
    
    # Run segmentation
    print("Running inference...")
    input_name = skyseg_session.get_inputs()[0].name
    output_name = skyseg_session.get_outputs()[0].name
    print(f"Model input name: {input_name}, output name: {output_name}")
    
    outputs = skyseg_session.run([output_name], {input_name: img_input})
    sky_pred = np.array(outputs).squeeze()
    print(f"Raw output shape: {sky_pred.shape}")
    print(f"Raw output - min: {sky_pred.min():.4f}, max: {sky_pred.max():.4f}, mean: {sky_pred.mean():.4f}")
    
    # Post-process: normalize to 0-255
    print("Post-processing...")
    min_val = np.min(sky_pred)
    max_val = np.max(sky_pred)
    sky_pred = (sky_pred - min_val) / (max_val - min_val) * 255
    sky_pred = sky_pred.astype(np.uint8)
    print(f"Normalized output - min: {sky_pred.min()}, max: {sky_pred.max()}, mean: {sky_pred.mean():.2f}")
    
    # Resize back to original resolution
    sky_mask_resized = cv2.resize(sky_pred, (img.shape[1], img.shape[0]))
    print(f"Resized mask shape: {sky_mask_resized.shape}")
    
    # Apply threshold
    threshold = 32
    frame_sky_mask = sky_mask_resized < threshold
    sky_points = np.sum(frame_sky_mask)
    total_points = img.shape[0] * img.shape[1]
    sky_percentage = (sky_points / total_points) * 100
    
    print(f"\n=== RESULTS ===")
    print(f"Threshold: {threshold}")
    print(f"Sky points (< {threshold}): {sky_points} out of {total_points} ({sky_percentage:.2f}%)")
    print(f"Non-sky points (>= {threshold}): {total_points - sky_points} ({100-sky_percentage:.2f}%)")
    
    # Save visualization
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_sky_mask.png"
    
    # Create visualization: original | mask | overlay
    mask_viz = np.stack([sky_mask_resized, sky_mask_resized, sky_mask_resized], axis=-1)
    
    # Create colored overlay (red for sky)
    overlay = img.copy()
    overlay[frame_sky_mask] = [0, 0, 255]  # Red for sky areas
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # Concatenate horizontally
    vis = np.hstack([img, mask_viz, blended])
    cv2.imwrite(str(output_path), vis)
    print(f"\nVisualization saved to: {output_path}")
    print("(Left: Original | Middle: Sky Mask | Right: Overlay with red=sky)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_sky_filter.py <image_path>")
        print("Example: python test_sky_filter.py test_image.jpg")
        sys.exit(1)
    
    test_sky_filter(sys.argv[1])
