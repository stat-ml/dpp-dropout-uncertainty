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

    # bins = np.concatenate((np.arange(0, 0.9, 0.1), np.arange(0.9, 1, 0.01)))
    bins = np.arange(0, 1, 0.1)

    for i, estimator in enumerate(record['estimators']):
        if estimator in ['max_entropy']:
            continue
        ue = record['uncertainties'][estimator]

        print(estimator)
        print(min(ue), max(ue))
        if args.group == 'bald_n':
            ue = ue / max(ue)

        for confidence_level in bins:
            point_confidences = 1 - ue
            bin_correct = is_correct[point_confidences > confidence_level]
            if len(bin_correct) > 0:
                accuracy = sum(bin_correct) / len(bin_correct)
            else:
                accuracy = None
            acc_conf.append((confidence_level, accuracy, estimator))
            count_conf.append((confidence_level, len(bin_correct), estimator))


plt.figure(figsize=(8, 6))
plt.title(f"Confidence-accuracy {args.name} {args.group}")
df = pd.DataFrame(acc_conf, columns=['Confidence level', 'Accuracy', 'Estimator'])
sns.lineplot('Confidence level', 'Accuracy', data=df, hue='Estimator')
plt.savefig(f"data/conf_accuracy_{args.name}_{args.group}", dpi=150)
plt.show()


plt.figure(figsize=(8, 6))
plt.title(f"Confidence-count {args.name} {args.group}")
df = pd.DataFrame(count_conf, columns=['Confidence level', 'Count', 'Estimator'])
sns.lineplot('Confidence level', 'Count', data=df, hue='Estimator')
plt.savefig(f"data/conf_count_{args.name}_{args.group}", dpi=150)
plt.show()

