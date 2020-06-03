import os
from shutil import copyfile



for ack in ['max_prob']:
    for i in range(3):
        file_from = f"logs/{ack}/cifar_{i}/ue.pickle"
        file_to = f"logs/classification/cifar_{i}/ue_{ack}.pickle"
        copyfile(file_from, file_to)
        print(file_from)
        print(file_to)

