# Uncertainty estimation via decorrelation and DPP
Code for paper "Dropout Strikes Back: Improved Uncertainty
Estimation via Diversity Sampling" by Evgenii Tsymbalov, Kirill Fedyanin, Maxim Panov

Main code with implemented methods (DPP, k-DPP, leverages masks for dropout) are in our [alpaca library](https://github.com/stat-ml/alpaca)

## Install dependency
```
pip install -r requirements.txt
```

## Regression
To get the experiment results from the paper, run the following notebooks
- `experiments/regression_1_big_exper_train-clean.ipynb` to train the models
- `experiments/regression_2_ll_on_trained_models.ipynb` to get the ll values for different datasets
- `experiments/regression_3_ood_w_training.ipynb` for the OOD experiments

## Classification

From the experiment folder run the following scripts. They goes in pairs, first script trains models and estimate the uncertainty, second just print the results.

#### Accuracy experiment on MNIST
```bash
python classification_ue.py mnist
python print_confidence_accuracy.py mnist
```
#### Accuracy experiment on CIFAR
```bash
python classification_ue.py cifar 
python print_confidence_accuracy.py cifar 
```
#### Accuracy experiment on ImageNet 
For the imagenet you need to manually download validation dataset (version ILSVRC2012) and put images to the `experiments/data/imagenet/valid` folder
```bash
python classification_imagenet.py 
python print_confidence_accuracy.py imagenet
```
#### OOD experiment on MNIST
```bash
python classification_ue_ood.py mnist
python print_ood.py mnist
```
#### OOD experiment on CIFAR 
```bash
python classification_ue_ood.py cifar 
python print_ood.py cifar 
```
#### OOD experiment on ImageNet 
```bash
python classification_imagenet.py --ood
python print_ood.py imagenet 
```

You can change the uncertainty estimation function for mnist/cifar by adding `-a=var_ratio` or `-a=max_prob` keys to the scripts.
