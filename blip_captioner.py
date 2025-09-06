
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import time
import os

class BLIPImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """Initialize BLIP Image Captioner"""
        print("Initializing BLIP Image Captioner...")
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = None
        self.load_model()
    
    def load_model(self):
        """Load BLIP model and processor"""
        try:
            print("Loading BLIP model components...")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            # Optimize for GPU if available
            if self.device.type == "cuda":
                self.model = self.model.half()
                print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("Model loaded on CPU")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_image(self, image_source):
        """Load image from file path or URL"""
        try:
            if isinstance(image_source, str):
                if image_source.startswith(('http://', 'https://')):
                    # Load from URL
                    response = requests.get(image_source)
                    image = Image.open(BytesIO(response.content))
                else:
                    # Load from file
                    if not os.path.exists(image_source):
                        raise FileNotFoundError(f"Image file not found: {image_source}")
                    image = Image.open(image_source)
            else:
                # Already a PIL Image
                image = image_source
            
            return image.convert('RGB')
            
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
    
    def generate_caption(self, image_source, prompt=None, **generation_kwargs):
        """Generate caption for an image"""
        start_time = time.time()
        
        try:
            # Load image
            print(f"Loading image: {image_source}")
            image = self.load_image(image_source)
            
            # Prepare inputs
            if prompt:
                print(f"Using prompt: '{prompt}'")
                inputs = self.processor(image, prompt, return_tensors="pt")
            else:
                print("🔄 Generating unconditional caption...")
                inputs = self.processor(image, return_tensors="pt")
            
            # Move to device
            inputs = inputs.to(self.device)
            if self.device.type == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}
            
            # Set generation parameters
            gen_kwargs = {
                'max_length': 50,
                'num_beams': 5,
                'early_stopping': True,
                'temperature': 0.7,
                'do_sample': True,
                'repetition_penalty': 1.1
            }
            gen_kwargs.update(generation_kwargs)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up conditional prompt
            if prompt and caption.lower().startswith(prompt.lower()):
                caption = caption[len(prompt):].strip()
            
            inference_time = time.time() - start_time
            print(f"Caption generated in {inference_time:.2f} seconds")
            
            return caption, image, inference_time
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            raise
    
    def display_result(self, image, caption, save_path=None):
        """Display image with caption"""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Caption: {caption}", fontsize=14, pad=20, wrap=True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Result saved to: {save_path}")
        
        plt.show()
    
    def save_caption(self, image_path, caption, output_file="captions.txt"):
        """Save caption to text file"""
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Caption: {caption}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
            
            print(f"Caption saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving caption: {e}")

