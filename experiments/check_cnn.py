#%%
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import seaborn as sns
from torchvision import datasets as datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from model_resnet import ResNet18, ResNet50, ResNet50_dropblock, ResNet50_dropchannel
from datasets import get_data


#%%
class Args:
    batch_size = 256
    train_batch_size = 5000
    lr = 1e-2
    gamma = 0.95
    log_interval = 100
    epochs = 2

args = Args()


#%%
loaders = get_data('svhn')
#%%
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#%%
train_loader = loaders['train']
test_loader = loaders['valid']
# model = Net()
model = ResNet50_dropchannel()
# model = ResNet50(input_size=3, dropout_rate=0.2)


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#%%
optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
#%%
predictions = []
labels = []
probs = []
with torch.no_grad():
    for x_batch, y_batch in  test_loader:
        logits = model(x_batch.cuda())
        probs.append(torch.max(torch.softmax(logits, dim=-1), dim=-1).values.cpu().numpy())
        predictions.append(torch.argmax(logits, dim=-1).cpu().numpy())
        labels.append(y_batch.numpy())
labels = np.concatenate(labels)
predictions = np.concatenate(predictions, axis=-1)
probs = np.concatenate(probs, axis=-1)

#%%
errors = labels != predictions

#%%
uq = 1 - probs

#%%
T = 200

# model.train()
model.eval()
predictions = []
for t in range(T):
    if (t+1) % 10  == 0:
        print(t)
    probs = []
    for x_batch, y_batch in  test_loader:
        with torch.no_grad():
            logits = model(x_batch.cuda(), dropout_mask=True, debug=True)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    predictions.append(np.concatenate(probs, axis=0))

#%%
predictions = np.array(predictions)
predictions.shape

#%%
def entropy(preds):
    return -np.mean((np.sum(np.log(preds) * preds, axis=-1)), axis=0)

def std_uq(preds):
    return np.std(np.max(preds, axis=-1), axis=0)

def bald(preds):
    means = np.mean(preds, axis=0)
    return - np.sum(np.log(means) * means, axis=-1) + entropy(preds)

def max_with_std(preds):
    stds = np.std(preds, axis=0)
    means = np.mean(preds, axis=0)
    mean_std = means - stds
    return 1 - np.max(mean_std, axis=-1)


acquisition = entropy
uq_mc = acquisition(predictions)
#%%
fpr, tpr, _ = roc_curve(errors, uq)
plt.plot(fpr, tpr)

fpr, tpr, _ = roc_curve(errors, uq_mc)
plt.plot(fpr, tpr)
plt.title("Resnet, Dropout 0.1/0.05, (max prob with std)")
#%%
aucs = []
nums = np.linspace(2, 500, 50)
max_score = 0
for num in nums:
    uq_sub = acquisition(predictions[:int(num)])
    score = roc_auc_score(errors, uq_sub)
    if score > max_score:
        i = num
        max_score = score
    aucs.append(score)
plt.plot(nums, aucs)
print(i)
#%%

stds = np.std(predictions, axis=0)
means = np.mean(predictions, axis=0)

#%%
df = pd.DataFrame({'errors':errors, 'stds': stds[:, 0]})
plt.figure(figsize=(10, 8))
sns.displot(df, x='stds', bins=20)
plt.title("Mc dropout std distribution")

#%%

#%%
uq_1 = entropy(predictions)
uq_2 = max_with_std(predictions)
plt.scatter(uq_1, uq_2)
#%%
mean_std = means - stds
args_1 = np.argmax(means, axis=-1)
args_2 = np.argmax(mean_std, axis=-1)
np.sum(args_1 != args_2)
#%%
