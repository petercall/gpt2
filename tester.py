from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split



data = pd.read_csv("data/subjects/test.csv")
print(data.shape[0])