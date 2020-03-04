# Uncertainty estimation via decorrelation and DPP

Code for paper "Dropout Strikes Back: Improved Uncertainty Estimation via Diversity Sampled Implicit Ensembles" by Evgenii Tsymbalov, Kirill Fedyanin and Maxim Panov 

 Our method improves Monte-Carlo dropout on inference for uncertainty estimation. Usually dropout masks are sampled randomly by Bernoulli distribution, we propose to sample them in a smart way, using decorrelation or determinantal point processes (DPP)

Main code is implemented in our [alpaca library](https://github.com/stat-ml/alpaca) for active learning and uncertainty estimation. This repository is for experiments themselves - to benchmark the performance and reproduce the results.

## Regression
#### Uncertainty regions and visualization 
Qualitative research. The idea is that uncertainty should be high in trained area and low for far regions.


#### Dolan-More curves
Series of experiments on few UCI datasets. We report performance for all experiments in one plot of [Dolan-More curve](https://abelsiqueira.github.io/blog/introduction-to-performance-profile/) of uncertainty accuracy.


## Classification
#### Error detection benchmark
Uncertainty estimation can be interpreted as an error detector. Thus we can treat it as prediction for binary task of correct/incorrect prediction. We run multiple times and report boxplots for ROC-AUC on MNIST/CIFAR/SVHN.
```
python -m experiments.classification_error_detection
```
#### OOD detection benchmark
Uncertainty estimation can be interpreted as an out-of-distribution samples detector. Thus we can treat it as prediction for binary task of in-distribution/out-of-distribution detection. We use two pairs of datasets: MNIST/Fashion-MNIST and CIFAR/SVHH. We run multiple times and report boxplots for ROC-AUC
```
python -m experiments.classification_ood_detection
```
#### Active Learning for computer vision
[Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) was run for computer vision on MNIST/CIFAR/SVHN. We report plot with error improvements.
```
python -m experiments.classification_active_learning
```
