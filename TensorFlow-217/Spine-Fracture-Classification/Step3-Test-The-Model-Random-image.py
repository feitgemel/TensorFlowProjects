"""
Spine Fracture Random Image Prediction Script

Description:
This script loads a trained CNN model and uses it to predict fractures in randomly
selected X-ray images. It includes random image selection, preprocessing, prediction,
and result visualization.

Purpose:
- Randomly select test images from the dataset
- Preprocess and make predictions using the trained model
- Compare predictions with true labels
- Visualize results
"""

# Import required libraries
import os                          # For file and directory operations
import numpy as np                 # For numerical operations
import tensorflow as tf            # For loading and using the trained model
import cv2                        # For image processing operations
import matplotlib.pyplot as plt    # For visualization
import random                     # For random image selection

# Configure paths for model and data
model_path = "D:/Data-Sets-Image-Classification/cervical fracture/model.keras"          # Path to final model
best_model_path = "D:/Data-Sets-Image-Classification/cervical fracture/best_model.keras"  # Path to best model (saved during training)
test_path = "D:/Data-Sets-Image-Classification/cervical fracture/test/"                 # Directory containing test data

# Define model input parameters
IMG_SIZE = (224,224)              # Required input size for the model

# Get class names from test directory
# This assumes directory names correspond to class names
class_names = sorted(os.listdir(test_path))
print("Available classes:", class_names)

# Load the best model (saved during training via callbacks)
model = tf.keras.models.load_model(best_model_path)
print("Model loaded successfully")
print(model.summary())            # Display model architecture

def predict_random_image():
    """
    Function to randomly select an image from the test dataset,
    make a prediction, and visualize the results.
    
    Steps:
    1. Randomly select a class and image
    2. Load and preprocess the image
    3. Make prediction using the model
    4. Display results
    """
    # Randomly select a class and image
    random_class = random.choice(class_names)                    # Select random class
    class_folder = os.path.join(test_path, random_class)        # Get path to class folder
    
    # Randomly select an image from the chosen class
    random_image = random.choice(os.listdir(class_folder))      # Select random image
    image_path = os.path.join(class_folder, random_image)       # Create full image path
    
    # Load and preprocess the image
    image = cv2.imread(image_path)                              # Load image in BGR format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         # Convert to RGB format
    
    # Image preprocessing steps:
    # 1. Resize to match model's expected input size
    resized_image = cv2.resize(image_rgb, IMG_SIZE)
    
    # 2. Normalize pixel values to range [0,1]
    normalized_image = resized_image/255.0
    print("Preprocessed image shape:", normalized_image.shape)   # Should be (224, 224, 3)
    
    # 3. Add batch dimension for model input
    input_array = np.expand_dims(normalized_image, axis=0)
    print("Model input shape:", input_array.shape)              # Should be (1, 224, 224, 3)
    
    # Make prediction
    predictions = model.predict(input_array)
    print("Raw predictions:", predictions)                      # Shows probability distribution
    
    # Process prediction results
    predicted_class_index = np.argmax(predictions)              # Get index of highest probability
    print("Predicted class index:", predicted_class_index)
    
    # Convert predicted index to class name
    predicted_class = class_names[predicted_class_index]
    print("Predicted class name:", predicted_class)
    
    # Visualize results
    plt.figure(figsize=(6,6))
    plt.imshow(image_rgb)                                       # Display the original RGB image
    plt.title(f"Predicted class: {predicted_class}\nTrue: {random_class}",
              fontsize=14)                                      # Show both prediction and true class
    plt.axis("off")                                            # Hide axes for cleaner visualization
    plt.show()

# Execute the random prediction function
predict_random_image()