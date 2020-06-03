import pickle
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from print_confidence_accuracy import estimator_name

#
parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('repeats', type=int)
parser.add_argument('--acquisition', '-a', type=str, default='bald')
parser.add_argument('--covariance', dest='covariance', action='store_true')
parser.add_argument('--resnet', dest='resnet', action='store_true')
args = parser.parse_args()

print(args)

count_conf = []
resnet_str = '_resnet' if args.resnet else ''
covariance_str = '_covar' if args.covariance else ''
acquisition_str = 'bald' if args.acquisition == 'bald_n' else args.acquisition

for i in range(args.repeats):
    file_name = f'logs/classification{resnet_str}/{args.name}_{i}/ue_ood_{acquisition_str}{covariance_str}.pickle'
    print(file_name)

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    # bins = np.concatenate((np.arange(0, 0.9, 0.1), np.arange(0.9, 1, 0.01)))
    # bins = np.arange(0, 1, 0.1)
    bins = np.concatenate((np.arange(0, 1, 0.1), [0.99]))
    for i, estimator in enumerate(record['estimators']):
        if estimator in ['max_entropy', 'ht_k_dpp', 'ht_decorrelating']:
            continue
        ue = record['uncertainties'][estimator]

        print(estimator)
        print(min(ue), max(ue))
        if args.acquisition == 'bald':
            ue = ue / max(ue)

        for confidence_level in bins:
            point_confidences = 1 - ue
            # bin_correct = is_correct[point_confidences > confidence_level]
            # if len(bin_correct) > 0:
            #     accuracy = sum(bin_correct) / len(bin_correct)
            # else:
            #     accuracy = None

            level_count = np.sum(point_confidences > confidence_level)
            count_conf.append((confidence_level, level_count, estimator_name(estimator)))
#
#
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.title(f"Confidence-accuracy {args.name} {args.acquisition}  {covariance_str}")
# df = pd.DataFrame(acc_conf, columns=['Confidence level', 'Accuracy', 'Estimator'])
# sns.lineplot('Confidence level', 'Accuracy', data=df, hue='Estimator')
# # plt.savefig(f"data/conf_accuracy_{args.name}_{args.acquisition}", dpi=150)
#
# plt.subplot(1, 2, 2)
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(9, 5))
plt.title(f"Confidence-count for OOD {args.name} {args.acquisition} {covariance_str}")
df = pd.DataFrame(count_conf, columns=['Confidence level', 'Count', 'Estimator'])
sns.lineplot('Confidence level', 'Count', data=df, hue='Estimator')
plt.subplots_adjust(right=0.7)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel(r"Number of samples, confidence > $\tau$")
plt.xlabel(r"$\tau$")
plt.savefig(f"data/conf_ood{resnet_str}_{args.name}_{args.acquisition}{covariance_str}", dpi=150)
plt.show()

