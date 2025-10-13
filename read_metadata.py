#!/usr/bin/env python3
"""
Quick script to read metadata from visualization images
"""

import sys
import json
from PIL import Image

def read_metadata(image_path):
    """Read and display metadata from a visualization image"""
    try:
        with Image.open(image_path) as img:
            print(f"\n{'='*80}")
            print(f"Metadata for: {image_path}")
            print(f"{'='*80}\n")
            
            if hasattr(img, 'info') and img.info:
                for key, value in img.info.items():
                    print(f"{key}:")
                    
                    # Try to parse JSON fields
                    if key == 'step_details':
                        try:
                            details = json.loads(value)
                            print(json.dumps(details, indent=2))
                        except:
                            print(f"  {value}")
                    # Show full content for important text fields
                    elif key in ['step_description', 'description']:
                        print(f"  {value}")
                    else:
                        # Pretty print with indentation
                        if len(str(value)) > 80:
                            print(f"  {str(value)[:80]}...")
                        else:
                            print(f"  {value}")
                    print()
            else:
                print("No metadata found in this image.")
                
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
            print(f"{'='*80}\n")
            
    except Exception as e:
        print(f"Error reading image: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_metadata.py <image_path>")
        print("\nExample:")
        print('  python read_metadata.py "visualizations/1ae2feb7/training_1/01_input.png"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    read_metadata(image_path)
