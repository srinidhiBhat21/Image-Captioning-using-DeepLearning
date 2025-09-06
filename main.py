#!/usr/bin/env python3
"""
BLIP Image Captioning - Main Execution Script
Run this file to start image captioning
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from blip_captioner import BLIPImageCaptioner

def check_images_folder():
    """Check if images folder exists and has images"""
    images_folder = Path("images")
    
    if not images_folder.exists():
        print("Creating images folder...")
        images_folder.mkdir()
        print(" Please add some images to the 'images' folder for testing")
        return []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_folder.glob(f"*{ext}"))
        image_files.extend(images_folder.glob(f"*{ext.upper()}"))
    
    return list(image_files)

def main():
    """Main execution function"""
    print("BLIP Image Captioning System")
    print("=" * 50)
    
    # Check for images
    image_files = check_images_folder()
    
    if not image_files:
        print(" No images found in the 'images' folder.")
        print("Please add some images (.jpg, .png, etc.) and run again.")
        
        # Option to use sample URL
        use_sample = input("\nWould you like to test with a sample image URL? (y/n): ")
        if use_sample.lower() == 'y':
            sample_url = "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=800"
            test_with_url(sample_url)
        return
    
    # Initialize captioner
    try:
        captioner = BLIPImageCaptioner()
        
        # Process images
        print(f"\nFound {len(image_files)} image(s)")
        
        for i, image_path in enumerate(image_files[:3]):  # Process first 3 images
            print(f"\n--- Processing Image {i+1}/{min(len(image_files), 3)} ---")
            print(f"File: {image_path.name}")
            
            # Generate caption
            caption, image, time_taken = captioner.generate_caption(str(image_path))
            
            print(f"Caption: {caption}")
            print(f"Time: {time_taken:.2f} seconds")
            
            # Display result
            captioner.display_result(image, caption)
            
            # Save caption
            captioner.save_caption(str(image_path), caption, "outputs/captions.txt")
            
            # Ask if user wants to continue
            if i < min(len(image_files), 3) - 1:
                continue_processing = input("\nProcess next image? (y/n): ")
                if continue_processing.lower() != 'y':
                    break
        
        print("\nImage captioning complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your setup and try again.")

def test_with_url(url):
    """Test with sample URL"""
    try:
        print(f"\nTesting with sample image from URL...")
        captioner = BLIPImageCaptioner()
        
        caption, image, time_taken = captioner.generate_caption(url)
        
        print(f"Caption: {caption}")
        print(f" Time: {time_taken:.2f} seconds")
        
        captioner.display_result(image, caption)
        
    except Exception as e:
        print(f"Error: {e}")

def interactive_mode():
    """Interactive mode for testing different prompts"""
    try:
        captioner = BLIPImageCaptioner()
        
        image_files = check_images_folder()
        if not image_files:
            print("No images available for interactive mode.")
            return
        
        image_path = str(image_files[0])
        print(f"Using image: {image_files[0].name}")
        
        prompts = [
            None,  # Unconditional
            "a photography of",
            "a painting of",
            "a sketch of",
            "an image of"
        ]
        
        print("\nTesting different prompts:")
        for prompt in prompts:
            prompt_text = "No prompt (unconditional)" if prompt is None else f"'{prompt}'"
            print(f"\nPrompt: {prompt_text}")
            
            caption, _, time_taken = captioner.generate_caption(image_path, prompt)
            print(f"Caption: {caption}")
            print(f"Time: {time_taken:.2f}s")
            
    except Exception as e:
        print(f"Error in interactive mode: {e}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run basic image captioning")
    print("2. Interactive prompt testing")
    print("3. Test setup verification")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        interactive_mode()
    elif choice == "3":
        # Import and run test
        from test_setup import test_imports, test_torch
        if test_imports():
            test_torch()
    else:
        print("Invalid choice. Running basic captioning...")
        main()
