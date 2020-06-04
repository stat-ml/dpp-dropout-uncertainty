from argparse import ArgumentParser
from pathlib import Path
import json
import ast
import pickle


parser = ArgumentParser()
parser.add_argument('--dataset-folder', type=str, default='data/imagenet')
args = parser.parse_args()

folder = Path(args.dataset_folder)
label_file =  folder / 'val.txt'
with open(label_file, 'r') as f:
    labels = list([int(line.split()[1]) for line in f.readlines()])

print(labels)
