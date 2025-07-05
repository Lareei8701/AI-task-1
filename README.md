# AI-task-1
## teachable Machine Model Predictor
A Python script to load and use your Teachable Machine trained models for image classification in Google Colab.
Overview
This project provides a complete solution for deploying Google's Teachable Machine models in Google Colab. It handles model loading, image preprocessing, and provides detailed predictions with confidence scores and visualizations.
## Prerequisites
1. Google Colab account
2. Teachable Machine model exported in TensorFlow → Keras format
3. Test images for classification
## Installation & Setup
# Step 1: Upload Files to Google Colab
Open Google Colab
Create a new notebook
Upload the following files to your Colab environment:
1. keras_model.h5 (your trained model)
2. labels.txt (class labels)
3. Your test images (jpg, png, etc.)
# Step 2: Install Dependencies
Run this command in a Colab cell:
python!pip install tensorflow==2.12.1 pillow numpy
# Step 3: Restart Runtime
After installing TensorFlow 2.12.1, restart the runtime:
Go to Runtime → Restart runtime
# Step 4: Run the Script
Copy and paste the main script into a new cell and run it.
Basic Usage
The script will automatically:
1. Check for your model files
2. Load the model and display class information
3. Find and process any images in the directory
4. Display predictions with confidence scores
