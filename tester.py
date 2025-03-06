from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

print()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.device(0))
print()