import pickle
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('repeats', type=int)
parser.add_argument('--group', type=str, default='ht')
args = parser.parse_args()


acc_conf = []
count_conf = []


for i in range(args.repeats):
    file_name = f'logs/{args.group}/{args.name}_{i}/ue.pickle'

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    prediction = np.argmax(record['probabilities'], axis=-1)
    is_correct = (prediction == record['y_val']).astype(np.int)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(args.group)

    for i, estimator in enumerate(record['estimators']):
        if estimator in ['max_entropy']:
            continue
        ue = record['uncertainties'][estimator]
        print(ue.shape)
        print(estimator)
        print(min(ue), max(ue))
        ax = plt.subplot(2, 3, i)
        ax.set_title(estimator)
        plt.hist(ue)

    plt.show()



