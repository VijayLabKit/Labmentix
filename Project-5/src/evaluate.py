import tensorflow as tf
from preprocess import get_datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_models():
    _, _, test_ds = get_datasets()
    
    def get_existing_path(model_name):
        keras_path = f'models/{model_name}.keras'
        h5_path = f'models/{model_name}.h5'
        return keras_path if os.path.exists(keras_path) else (h5_path if os.path.exists(h5_path) else None)

    custom_path = get_existing_path('custom_bird_drone')
    transfer_path = get_existing_path('transfer_bird_drone')

    if not custom_path or not transfer_path:
        print("Error: Models not found in 'models/' folder.")
        return

    print(f"\nLoading models...")
    custom_model = tf.keras.models.load_model(custom_path)
    transfer_model = tf.keras.models.load_model(transfer_path)

    models = {"Custom CNN": custom_model, "Transfer Learning": transfer_model}

    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    for name, model in models.items():
        print(f"\n=== EVALUATING: {name} ===")
        y_pred_probs = model.predict(test_ds)
        y_pred = (y_pred_probs > 0.5).astype(int)

        print(classification_report(y_true, y_pred, target_names=['Bird', 'Drone']))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bird', 'Drone'], yticklabels=['Bird', 'Drone'])
        plt.title(f'Confusion Matrix: {name}')
        plt.show()

if __name__ == "__main__":
    evaluate_models()