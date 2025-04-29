import os
import shutil

dir = "./dataset/train/fire/"
newdir = "./dataset/train/fire-cleaned/"

for name in os.listdir(dir):
    if "TRUE_COLOR" in name:
        shutil.copy(f"{dir}{name}", newdir)