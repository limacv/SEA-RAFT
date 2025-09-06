"""
Demo script for SEA-RAFT
"""

import argparse
import cv2
import torch
import time
import imageio
import os
import sys

# Add the parent directory to path to access core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from searaft import SEA_RAFT, remap
from core.utils.flow_viz import flow_to_image


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='SEA-RAFT Demo')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--image1', default='./custom/image1.jpg', help='Path to first image')
    parser.add_argument('--image2', default='./custom/image2.jpg', help='Path to second image')
    parser.add_argument('--output_dir', default='./custom', help='Output directory for results')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale factor for input images')
    parser.add_argument('--iters', type=int, default=4, help='Number of refinement iterations')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found!")
        return
    
    # Check if images exist
    if not os.path.exists(args.image1):
        print(f"Error: Image file {args.image1} not found!")
        return
    if not os.path.exists(args.image2):
        print(f"Error: Image file {args.image2} not found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading SEA-RAFT model from {args.checkpoint}")
    raft = SEA_RAFT(args.checkpoint)
    
    # Load images
    print(f"Loading images: {args.image1}, {args.image2}")
    image1 = cv2.imread(args.image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(args.image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Convert to tensors
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    image1 = image1[None]
    image2 = image2[None]
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    image1 = image1.to(device)
    image2 = image2.to(device)
    raft = raft.to(device)
    
    # Run inference
    print("Running inference...")
    start = time.time()
    flow, info = raft(image1, image2, iters=args.iters, scale=args.scale)
    end = time.time()
    print(f"Inference time: {end - start:.4f}s")
    
    # Save flow visualization
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    flow_output_path = os.path.join(args.output_dir, "flow.jpg")
    cv2.imwrite(flow_output_path, flow_vis)
    print(f"Flow visualization saved to: {flow_output_path}")
    
    # Create remapped image
    flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    remapped_image = remap(image2, flow)
    remap_output_path = os.path.join(args.output_dir, "remapped.jpg")
    imageio.imwrite(remap_output_path, remapped_image[0].permute(1, 2, 0).cpu().numpy().astype('uint8'))
    print(f"Remapped image saved to: {remap_output_path}")
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
