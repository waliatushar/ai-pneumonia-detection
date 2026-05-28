import os
from PIL import Image

# 🔥 paths of your dataset
folders = ["dataset/train", "dataset/test"]

size = (224,224)

for folder in folders:
    print(f"\nProcessing folder: {folder}")

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(subdir, file)

            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(size)
                img.save(path)

            except:
                print("❌ Error in image:", path)

print("\n✅ ALL TRAIN + TEST IMAGES RESIZED TO 224x224")