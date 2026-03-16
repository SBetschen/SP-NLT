import os
import random
import shutil

# Paths
source_folder = "data"      # folder with all images
test_folder = "data/test"          # where the 200 test images will go

# Create test folder if it doesn't exist
os.makedirs(test_folder, exist_ok=True)

# List all tif images
images = [f for f in os.listdir(source_folder) if f.endswith(".tif")]

# Check that there are enough images
if len(images) < 200:
    raise ValueError("Not enough images in the folder.")

# Randomly select 200 images
test_images = random.sample(images, 200)

# Move selected images
for img in test_images:
    src = os.path.join(source_folder, img)
    dst = os.path.join(test_folder, img)
    shutil.move(src, dst)

print("Moved", len(test_images), "images to the test folder.")