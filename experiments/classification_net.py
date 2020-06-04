import os
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

# datasets.ImageFolder


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()

image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        files = sorted(os.listdir(folder))
        self.files = [file for file in files if file.endswith(".JPEG")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.folder / self.files[idx]
        image = Image.open(img_path).convert('RGB')

        image = image_transforms(image)

        return image


def predict_valid():
    parser = ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, default='data/imagenet')
    parser.add_argument('--bs', type=int, default=64)
    args = parser.parse_args()

    label_file = Path(args.dataset_folder)/'val.txt'

    with open(label_file, 'r') as f:
        labels = list([int(line.split()[1]) for line in f.readlines()])

    valid_folder = Path(args.dataset_folder) / 'valid'

    dataset = ImageDataset(valid_folder)
    loader = DataLoader(dataset, batch_size=args.bs)

    model = models.resnet18(pretrained=True)
    model.eval().cuda()

    results = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print((i + 1) * args.bs)
            probabilities = torch.softmax(model(batch.cuda()), dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            results.extend(list(list(predictions.cpu().numpy())))
    print(results)
    print(accuracy_score(results, labels))


if __name__ == '__main__':
    predict_valid()




plt.scatter()


