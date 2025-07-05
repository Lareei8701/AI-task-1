# Install required packages (run this cell first in Google Colab)
!pip install tensorflow==2.12.1 pillow numpy

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
from google.colab import files
import matplotlib.pyplot as plt

class TeachableMachinePredictor:
    def __init__(self, model_path, labels_path):
        """
        Initialize the predictor with model and labels
        
        Args:
            model_path: Path to the keras_model.h5 file
            labels_path: Path to the labels.txt file
        """
        self.model = keras.models.load_model(model_path, compile=False)
        self.labels = self.load_labels(labels_path)
        
    def load_labels(self, labels_path):
        """Load class labels from labels.txt file"""
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for Teachable Machine model
        Teachable Machine typically expects 224x224 images normalized to [0,1]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224 (standard for Teachable Machine)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0,1]
        image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image_path, show_image=True):
        """
        Make prediction on an image
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.labels[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Show image if requested
        if show_image:
            img = Image.open(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f'Predicted: {predicted_class} (Confidence: {confidence:.2%})')
            plt.axis('off')
            plt.show()
        
        # Create results dictionary
        results = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': {}
        }
        
        # Add all class probabilities
        for i, label in enumerate(self.labels):
            results['all_predictions'][label] = predictions[0][i]
        
        return results
    
    def predict_top_n(self, image_path, n=3):
        """
        Get top N predictions for an image
        
        Args:
            image_path: Path to the image file
            n: Number of top predictions to return
        
        Returns:
            List of tuples (class_name, confidence)
        """
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image)
        
        # Get top N predictions
        top_indices = np.argsort(predictions[0])[::-1][:n]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.labels[idx]
            confidence = predictions[0][idx]
            top_predictions.append((class_name, confidence))
        
        return top_predictions

# Main execution code
def main():
    print("=== Teachable Machine Model Predictor ===\n")
    
    # Check if model files exist
    if not os.path.exists('keras_model.h5'):
        print("Error: keras_model.h5 not found! Please upload it to the notebook first.")
        return
    
    if not os.path.exists('labels.txt'):
        print("Error: labels.txt not found! Please upload it to the notebook first.")
        return
    
    # Initialize predictor
    print("Loading model...")
    predictor = TeachableMachinePredictor('keras_model.h5', 'labels.txt')
    print(f"Model loaded successfully!")
    print(f"Number of classes: {len(predictor.labels)}")
    print(f"Classes: {predictor.labels}")
    
    # List available image files in the current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    available_images = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not available_images:
        print("\nNo image files found in the current directory.")
        print("Please upload some images and run again, or use the upload option below.")
        print("\nAlternatively, upload an image now:")
        test_images = files.upload()
        available_images = list(test_images.keys())
    else:
        print(f"\nFound {len(available_images)} image(s) in the directory:")
        for i, img in enumerate(available_images, 1):
            print(f"{i}. {img}")
        
        # Option to upload additional images
        print("\nWould you like to upload additional images? (Press Enter to skip)")
        additional_images = files.upload()
        if additional_images:
            available_images.extend(additional_images.keys())
    
    # Make predictions on all available images
    for filename in available_images:
        print(f"\n--- Predicting for {filename} ---")
        
        try:
            # Single prediction
            result = predictor.predict(filename)
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            # Top 3 predictions
            print(f"\nTop 3 predictions:")
            top_predictions = predictor.predict_top_n(filename, n=3)
            for i, (class_name, confidence) in enumerate(top_predictions, 1):
                print(f"{i}. {class_name}: {confidence:.2%}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
        
        print("-" * 50)

# Alternative function for direct use with file paths
def predict_image(model_path, labels_path, image_path):
    """
    Direct function to predict a single image
    Useful when you already have files uploaded
    """
    predictor = TeachableMachinePredictor(model_path, labels_path)
    return predictor.predict(image_path)

# Run the main function
if __name__ == "__main__":
    main()

# Example usage after files are uploaded:
# predictor = TeachableMachinePredictor('keras_model.h5', 'labels.txt')
# result = predictor.predict('your_image.jpg')
# print(f"Prediction: {result['predicted_class']} with {result['confidence']:.2%} confidence")