import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/al/svhn_strong_conv_7000_50.csv', names=['id', 'accuracy', 'step', 'method'], index_col='id', skiprows=1)


plt.title('SVHN')
sns.lineplot('step', 'accuracy', hue='method', data=df)
plt.show()


