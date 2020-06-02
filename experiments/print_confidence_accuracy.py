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
parser.add_argument('--acquisition', '-a', type=str, default='bald')
parser.add_argument('--covariance', dest='covariance', action='store_true')
parser.add_argument('--resnet', dest='resnet', action='store_true')
args = parser.parse_args()


acc_conf = []
count_conf = []

covariance_str = '_covar' if args.covariance else ''
resnet_str = '_resnet' if args.resnet else ''
acquisition_str = 'bald' if args.acquisition == 'bald_n' else args.acquisition

for i in range(args.repeats):
    file_name = f'logs/classification{resnet_str}/{args.name}_{i}/ue_{acquisition_str}{covariance_str}.pickle'
    print(file_name)

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
        if args.acquisition == 'bald':
            ue = ue / max(ue)

        for confidence_level in bins:
            point_confidences = 1 - ue
            bin_correct = is_correct[point_confidences > confidence_level]
            if len(bin_correct) > 0:
                accuracy = sum(bin_correct) / len(bin_correct)
            else:
                accuracy = None

            if estimator == 'max_prob':
                estimator_name = estimator
            else:
                estimator_name = f"{estimator}_{args.acquisition}"

            acc_conf.append((confidence_level, accuracy, estimator_name))
            count_conf.append((confidence_level, len(bin_correct), estimator_name))


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.title(f"Confidence-accuracy {args.name} {args.acquisition}  {covariance_str}")
df = pd.DataFrame(acc_conf, columns=['Confidence level', 'Accuracy', 'Estimator'])
sns.lineplot('Confidence level', 'Accuracy', data=df, hue='Estimator')
# plt.savefig(f"data/conf_accuracy_{args.name}_{args.acquisition}", dpi=150)

plt.subplot(1, 2, 2)
# plt.figure(figsize=(8, 6))
plt.title(f"Confidence-count {args.name} {args.acquisition} {covariance_str}")
df = pd.DataFrame(count_conf, columns=['Confidence level', 'Count', 'Estimator'])
sns.lineplot('Confidence level', 'Count', data=df, hue='Estimator')
plt.savefig(f"data/conf{resnet_str}_accuracy_{args.name}_{args.acquisition}{covariance_str}", dpi=150)
# plt.savefig(f"data/conf_count_{args.name}_{args.acquisition}", dpi=150)
plt.show()

