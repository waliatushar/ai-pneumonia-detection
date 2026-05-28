from PIL import Image
import os

folder = "dataset/train"   # change if needed

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.startswith("."):
            continue

        path = os.path.join(root, file)
        img = Image.open(path)

        print(file, "→", img.size)