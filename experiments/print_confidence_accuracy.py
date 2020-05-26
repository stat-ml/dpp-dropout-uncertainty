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
args = parser.parse_args()

acc_conf = []
count_conf = []
for i in range(args.repeats):
    file_name = f'logs/var_ratio/{args.name}_{i}/ue.pickle'

    with open(file_name, 'rb') as f:
        record = pickle.load(f)

    prediction = np.argmax(record['probabilities'], axis=-1)
    is_correct = (prediction == record['y_val']).astype(np.int)

    bins = np.arange(0, 1, 0.1)

    for estimator in record['estimators']:
        if estimator in ['max_entropy']:
            continue
        ue = record['uncertainties'][estimator]
        for confidence_level in bins:
            point_confidences = 1 - ue
            bin_correct = is_correct[point_confidences > confidence_level]
            acc_conf.append((confidence_level, sum(bin_correct) / len(bin_correct), estimator))
            count_conf.append((confidence_level, len(bin_correct), estimator))


plt.figure(figsize=(8, 6))
plt.title(f"Confidence-accuracy {args.name}")
print(acc_conf)
df = pd.DataFrame(acc_conf, columns=['Confidence level', 'Accuracy', 'Estimator'])
sns.lineplot('Confidence level', 'Accuracy', data=df, hue='Estimator')
plt.savefig(f"data/conf_accuracy_{args.name}_vr", dpi=150)
plt.show()



plt.figure(figsize=(8, 6))
plt.title(f"Confidence-count {args.name}")
print(count_conf)
df = pd.DataFrame(count_conf, columns=['Confidence level', 'Count', 'Estimator'])
sns.lineplot('Confidence level', 'Count', data=df, hue='Estimator')
plt.savefig(f"data/conf_count_{args.name}_vr", dpi=150)
plt.show()


































#
#
# # df = pd.read_csv('data/al/ht_mnist_200_10.csv', names=['id', 'accuracy', 'step', 'method'], index_col='id', skiprows=1)
# #
# #
# # plt.figure(figsize=(8, 6))
# # plt.title('MNIST')
# # sns.lineplot('step', 'accuracy', hue='method', data=df)
# # plt.savefig('data/al/al_ht_mnist_200_10')
# # plt.show()
#
#
# config = {
#     'name': 'cifar'
# }
# # df = pd.DataFrame(rocaucs, columns=['Estimator', 'ROC-AUCs'])
# df = pd.read_csv(f"logs/{config['name']}_ed.csv", names=['id', 'ROC-AUCs', 'Estimator'], index_col='id', skiprows=1)
# plt.figure(figsize=(9, 6))
# sns.boxplot('Estimator', 'ROC-AUCs', data=df)
# plt.title(f"Error detection for {config['name']}")
# plt.savefig(f"data/ed/{config['name']}.png", dpi=150)
# plt.show()
#
# df.to_csv(f"logs/{config['name']}_ed.csv")
