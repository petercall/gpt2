# from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import AutoTokenizer
# import torch.optim.lr_scheduler as lr_scheduler
# import torch.optim as optim
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sys
import os
# from tqdm import tqdm
import pandas as pd

location = "data/subjects"
names = ["train", "validation", "test"]

for name in names:
    in_file = os.path.join(location, name + ".csv")
    data = pd.read_csv(in_file) 
    text = data["model_output"].str.cat(sep = " ")
    out_file = os.path.join(location, name + ".txt")
    with open(out_file, 'w') as my_file:
        my_file.write(text)


