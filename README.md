#!/bin/bash

# Clone the repository (replace with your actual repository URL)
echo "Cloning the repository..."
git clone https://github.com/yourusername/vqa-webcam-image-upload.git
cd vqa-webcam-image-upload || { echo "Repository clone failed"; exit 1; }

# Install the required Python packages
echo "Installing required Python packages..."
pip install torch transformers opencv-python pillow requests || { echo "Package installation failed"; exit 1; }

# Download the BLIP model weights
echo "Downloading BLIP model weights..."
python3 -c "
from transformers import BlipProcessor, BlipForQuestionAnswering

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')
" || { echo "Model download failed"; exit 1; }

# Run the VQA system
echo "Running the VQA system..."
python3 vqa_system.py || { echo "Failed to run the VQA system"; exit 1; }

echo "Setup complete. The VQA system is now running."
