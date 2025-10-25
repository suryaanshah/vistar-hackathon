import cv2
import matplotlib.pyplot as plt
import os
import random

# --- Configuration ---
# Update this path to where you unzipped the data
data_dir = 'archive/chest_xray' 
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Categories
categories = ['NORMAL', 'PNEUMONIA']

# --- Function to display a random image ---
def show_random_image(category):
    """Loads and displays a random image from the specified category (NORMAL or PNEUMONIA)"""
    
    path = os.path.join(train_dir, category)
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}")
        return

    # Get a random image from the folder
    filename = random.choice(os.listdir(path))
    img_path = os.path.join(path, filename)
    
    # Read the image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale
    
    if img is None:
        print(f"Error: Could not read image: {img_path}")
        return

    # Display the image using Matplotlib
    plt.imshow(img, cmap='gray')
    plt.title(f'Category: {category}\nFile: {filename}')
    plt.axis('off') # Hide axes
    plt.show()

# --- Let's look at the data ---
print("Displaying a NORMAL X-Ray...")
show_random_image('NORMAL')

print("Displaying a PNEUMONIA X-Ray...")
show_random_image('PNEUMONIA')

# --- Let's count the data ---
print("\n--- Data Counts ---")
for category in categories:
    train_count = len(os.listdir(os.path.join(train_dir, category)))
    test_count = len(os.listdir(os.path.join(test_dir, category)))
    print(f"Category: {category}")
    print(f"  Training images: {train_count}")
    print(f"  Test images:     {test_count}")
    print("-" * 20) 