#!/usr/bin/env python3
"""
BLIP Image Captioning Backend API
Flask server for handling image captioning requests using BLIP model
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
import os
import io
import base64
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model (loaded once at startup)
processor = None
model = None
device = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_blip_model():
    """Load BLIP model and processor (called once at startup)"""
    global processor, model, device
    
    try:
        logger.info("Loading BLIP model components...")
        
        # Load processor and model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Setup device (GPU if available, CPU otherwise)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Optimize for GPU if available
        if device.type == "cuda":
            model = model.half()  # Use half precision for faster inference
            logger.info(f" BLIP model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("BLIP model loaded on CPU")
            
        return True
        
    except Exception as e:
        logger.error(f" Error loading BLIP model: {e}")
        return False

def generate_caption_from_image(image, prompt=None):
    """Generate caption for a given PIL image"""
    try:
        start_time = time.time()
        
        # Prepare inputs based on whether prompt is provided
        if prompt and prompt.strip():
            inputs = processor(image, prompt.strip(), return_tensors="pt").to(device)
            logger.info(f"Using prompt: '{prompt.strip()}'")
        else:
            inputs = processor(image, return_tensors="pt").to(device)
            logger.info("Generating unconditional caption...")
        
        # Handle GPU precision
        if device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        # Generate caption using BLIP model
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        # Decode the generated caption
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Clean up prompt from result if it appears at the beginning
        if prompt and prompt.strip() and caption.lower().startswith(prompt.strip().lower()):
            caption = caption[len(prompt.strip()):].strip()
        
        # Calculate inference time
        inference_time = time.time() - start_time
        logger.info(f"Caption generated in {inference_time:.2f} seconds")
        
        return {
            'caption': caption,
            'inference_time': round(inference_time, 2),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return {
            'error': f'Error generating caption: {str(e)}',
            'success': False
        }

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status"""
    return jsonify({
        'status': 'healthy',
        'device': str(device) if device else 'unknown',
        'model_loaded': processor is not None and model is not None,
        'timestamp': time.time()
    })

@app.route('/api/caption', methods=['POST'])
def generate_caption_api():
    """Main API endpoint for generating image captions"""
    try:
        # Check if model is loaded
        if processor is None or model is None:
            return jsonify({
                'error': 'BLIP model not loaded. Please restart the server.',
                'success': False
            }), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided in the request',
                'success': False
            }), 400
        
        image_file = request.files['image']
        
        # Check if a file was actually selected
        if image_file.filename == '':
            return jsonify({
                'error': 'No image file selected',
                'success': False
            }), 400
        
        # Check file extension
        if not allowed_file(image_file.filename):
            return jsonify({
                'error': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400
        
        # Get optional prompt
        prompt = request.form.get('prompt', '').strip()
        
        # Process the image
        try:
            # Read image file
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Log image info
            logger.info(f"Processing image: {image_file.filename} ({image.size[0]}x{image.size[1]})")
            
        except Exception as e:
            return jsonify({
                'error': f'Invalid image file: {str(e)}',
                'success': False
            }), 400
        
        # Generate caption
        result = generate_caption_from_image(image, prompt)
        
        if result['success']:
            # Log successful generation
            logger.info(f"Generated caption: '{result['caption']}'")
            
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in caption API: {e}")
        return jsonify({
            'error': f'Unexpected server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/models/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_name': 'Salesforce/blip-image-captioning-base',
        'device': str(device) if device else 'unknown',
        'model_loaded': processor is not None and model is not None,
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size_mb': MAX_CONTENT_LENGTH // (1024 * 1024)
    })

# Serve static files (for development only)
@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('..', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('..', path)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': f'File too large. Maximum size is {MAX_CONTENT_LENGTH // (1024 * 1024)}MB',
        'success': False
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

# Application startup
def initialize_app():
    """Initialize the application"""
    logger.info("🎬 Starting BLIP Image Captioning Server...")
    
    # Load BLIP model
    if not load_blip_model():
        logger.error("Failed to load BLIP model. Exiting...")
        return False
    
    logger.info("Server initialized successfully!")
    logger.info("Available endpoints:")
    logger.info("   - POST /api/caption - Generate image captions")
    logger.info("   - GET /health - Health check")
    logger.info("   - GET /api/models/info - Model information")
    
    return True

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        # Start the Flask development server
        logger.info("Starting Flask server...")
        logger.info("Frontend will be available at: http://localhost:5000")
        logger.info("API endpoint: http://localhost:5000/api/caption")
        logger.info("Press Ctrl+C to stop the server")
        
        app.run(
            debug=True,          # Enable debug mode for development
            host='0.0.0.0',      # Listen on all interfaces
            port=5000,           # Port number
            threaded=True        # Handle multiple requests concurrently
        )
    else:
        logger.error("❌ Failed to initialize application")
        exit(1)