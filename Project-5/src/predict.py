import tensorflow as tf
import numpy as np
import os

def get_model_path():
    keras_path = r'models/transfer_bird_drone.keras'
    h5_path = r'models/transfer_bird_drone.h5'
    if os.path.exists(keras_path):
        return keras_path
    elif os.path.exists(h5_path):
        return h5_path
    return None

def predict_image(img_path):
    model_path = get_model_path()
 
    if not model_path:
        print("Error: Model file not found in 'models/' folder. Please ensure transfer_bird_drone.h5 or .keras exists.")
        return

    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    print(f"\nUsing model: {model_path}")
    print(f"Analyzing image: {img_path}...")

    model = tf.keras.models.load_model(model_path, compile=False)

    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    img_array = img_array / 255.0          

    # 3. Make Prediction
    prediction = model.predict(img_array, verbose=0)
    score = prediction[0][0]

    
    if score > 0.5:
        print(f"Result: DRONE (Confidence: {score*100:.2f}%)")
    else:
        print(f"Result: BIRD (Confidence: {(1-score)*100:.2f}%)")

if __name__ == "__main__":
  
    bird_test_dir = r"C:\Users\ishan\OneDrive\College\Project\dataset\test\bird"
    
    if os.path.exists(bird_test_dir):
        # Get list of files and pick the first one
        files = [f for f in os.listdir(bird_test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            target_image = os.path.join(bird_test_dir, files[0])
            predict_image(target_image)
        else:
            print(f"No images found in {bird_test_dir}")
    else:
        print("--- Manual Prediction Mode ---")
        user_path = input("Enter the full path to an image file: ").strip().replace('"', '')
        predict_image(user_path)