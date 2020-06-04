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

resnet_str = '_resnet' if args.resnet else ''
covariance_str = '_covar' if args.covariance else ''
acquisition_str = 'bald' if args.acquisition == 'bald_n' else args.acquisition


def process_file(file_name, count_conf):
    print(file_name)

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    bins = np.concatenate((np.arange(0, 1, 0.1), [0.99]))
    for i, estimator in enumerate(record['estimators']):
        if estimator not in ['mc_dropout', 'max_prob', 'ht_dpp', 'ensemble_max_prob']:
            continue
        ue = record['uncertainties'][estimator]

        print(estimator)
        print(min(ue), max(ue))
        if args.acquisition == 'bald':
            ue = ue / max(ue)

        for confidence_level in bins:
            point_confidences = 1 - ue
            level_count = np.sum(point_confidences > confidence_level)
            count_conf.append((confidence_level, level_count, estimator_name(estimator)))


count_conf = []

for i in range(args.repeats):
    file_name = f'logs/classification{resnet_str}/{args.name}_{i}/ue_ood_{acquisition_str}{covariance_str}.pickle'
    process_file(file_name, count_conf)
    file_name = f'logs/classification/{args.name}_{i}/ue_ensemble_ood.pickle'
    process_file(file_name, count_conf)



plt.rcParams.update({'font.size': 14})
plt.rc('grid', linestyle="--")
plt.figure(figsize=(9, 5))
plt.title(f"Confidence-count for OOD {args.name} {args.acquisition} {covariance_str}")
df = pd.DataFrame(count_conf, columns=['Confidence level', 'Count', 'Estimator'])
sns.lineplot('Confidence level', 'Count', data=df, hue='Estimator')
plt.subplots_adjust(right=0.7)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel(r"Number of samples, confidence > $\tau$")
plt.xlabel(r"$\tau$")
plt.grid()
plt.savefig(f"data/conf_ood{resnet_str}_{args.name}_{args.acquisition}{covariance_str}", dpi=150)
plt.show()

