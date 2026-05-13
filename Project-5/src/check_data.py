import os

base_path = r"C:\Users\ishan\OneDrive\College\Project\dataset"

def check_folders():
    folders = ['train', 'valid', 'test']
    classes = ['bird', 'drone']
    
    print(f"--- Checking Dataset at: {base_path} ---")
    
    if not os.path.exists(base_path):
        print(f"ERROR: The path {base_path} does not exist.")
        return

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            print(f"\nFolder found: {folder}")
            for cls in classes:
                class_path = os.path.join(folder_path, cls)
                if os.path.exists(class_path):
                    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                    print(f"  - {cls}: {len(files)} images")
                else:
                    print(f"  - ERROR: Subfolder {cls} missing in {folder}")
        else:
            print(f"ERROR: Folder {folder} missing at {folder_path}")

if __name__ == "__main__":
    check_folders()