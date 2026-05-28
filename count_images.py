import os

normal_path = "dataset/train/normal"
pneumonia_path = "dataset/train/pneumonia"
normal1_path = "dataset/test/normal"
pneumonia1_path = "dataset/test/pneumonia"


normal_count = len(os.listdir(normal_path))
pneumonia_count = len(os.listdir(pneumonia_path))
normal1_count = len(os.listdir(normal1_path))
pneumonia1_count = len(os.listdir(pneumonia1_path))

print("Normal images:", normal_count)
print("Pneumonia images:", pneumonia_count)
print("Normal images : ", normal1_count)
print("Pneumonia images : ", pneumonia1_count)