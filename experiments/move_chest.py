import os
from pathlib import Path
from shutil import copy

from PIL import Image

## The script to make Chexpert directory style similar to the ImageNet
root_dir = Path('data/chest')
source = root_dir / 'CheXpert-v1.0-small' / 'train'
target = root_dir / 'train'


folder_list = sorted(os.listdir(source))

for folder in folder_list:
    with_images = source/folder/'study1'

    if os.path.exists(with_images / 'view1_frontal.jpg'):
        file_path = with_images / 'view1_frontal.jpg'
    else:
        file_path = with_images / 'view1_lateral.jpg'
    image = Image.open(file_path)

    copy(file_path, target/f"{folder}.jpg")

    print(image.size, folder)



