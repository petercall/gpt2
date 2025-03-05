import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device("cuda") if torch.cuda.is_availble() else torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

