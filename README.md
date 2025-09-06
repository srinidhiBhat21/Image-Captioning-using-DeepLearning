# Image-Captioning-using-DeepLearning
Pretrained Deeplearning model is used to analyze the image and output the caption as "What is happening" in the image uploaded. 

Model Setup:
Loads the BLIP image captioning model and processor, with automatic device selection for CUDA/GPU acceleration if available .

Image Loading:
Includes a loader capable of handling both local file paths and remote image URLs, converting images to a suitable RGB format for inference .

Caption Generation:
Implements a function to generate natural language descriptions for images, supporting optional prompt-driven (conditional) or unconditional captioning. Inference time is recorded for performance benchmarking .

Visualization:
Utilizes Matplotlib to display images alongside their generated captions for qualitative review, and supports batch processing with grid-based visualization for multiple images .

Application Context
This code streamlines the process from image acquisition (local or online), through AI-powered caption generation, to professional result presentation. It demonstrates proficiency with PyTorch, Hugging Face Transformers, computer vision, and deep learning deployment pipelines, making it suitable as a showcase project for roles in AI, computer vision, or machine learning engineering .
