
import os
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample


mod = PyroModule[nn.Linear](10, 3)

print(mod)

class Some:
    def __getitem__(self, x):
        if x==1:
            return 2
        return 0

# some = Some)
print(Some[1])
