import os
import matplotlib.pyplot as plt
from PIL import Image

# Updated path to match your actual directory
base_path = r"C:\Users\ishan\OneDrive\College\Project\dataset"

def analyze():
    train_path = os.path.join(base_path, 'train')
    classes = ['bird', 'drone']
    
    if not os.path.exists(train_path):
        print(f"ERROR: Cannot find train folder at {train_path}")
        return

    # 1. Check Class Balance
    counts = {}
    for cls in classes:
        class_dir = os.path.join(train_path, cls)
        if os.path.exists(class_dir):
            counts[cls] = len(os.listdir(class_dir))
        else:
            print(f"Warning: Folder for {cls} not found.")
    
    print(f"Class Distribution: {counts}")
    
    # 2. Show Sample Images
    plt.figure(figsize=(10, 5))
    for i, cls in enumerate(classes):
        class_folder = os.path.join(train_path, cls)
        # Get list of images and pick the first one
        img_list = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if img_list:
            img_path = os.path.join(class_folder, img_list[0])
            img = Image.open(img_path)
            plt.subplot(1, 2, i+1)
            plt.imshow(img)
            plt.title(f"Sample: {cls}\nSize: {img.size}")
            plt.axis('off')
    
    print("Opening plot window... (Close it to continue)")
    plt.show()

if __name__ == "__main__":
    analyze()