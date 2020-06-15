import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# df = pd.read_csv('data/al/ht_mnist_200_10.csv', names=['id', 'accuracy', 'step', 'method'], index_col='id', skiprows=1)
#
#
# plt.figure(figsize=(8, 6))
# plt.title('MNIST')
# sns.lineplot('step', 'accuracy', hue='method', data=df)
# plt.savefig('data/al/al_ht_mnist_200_10')
# plt.show()


config = {
    'name': 'cifar'
}
# df = pd.DataFrame(rocaucs, columns=['Estimator', 'ROC-AUCs'])
df = pd.read_csv(f"logs/{config['name']}_ed.csv", names=['id', 'ROC-AUCs', 'Estimator'], index_col='id', skiprows=1)
plt.figure(figsize=(9, 6))
sns.boxplot('Estimator', 'ROC-AUCs', data=df)
plt.title(f"Error detection for {config['name']}")
plt.savefig(f"data/ed/{config['name']}.png", dpi=150)
plt.show()

df.to_csv(f"logs/{config['name']}_ed.csv")
