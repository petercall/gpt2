import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np


data = pd.read_csv("data/subjects/train.csv")
print(round(data["subject"].value_counts() * 100 / data.shape[0], 2))











# train = pd.read_csv("data/subjects/train.csv")

# MODEL_NAME = "facebook/bart-large-mnli"

# TARGET_CLASSES = ["mathematics", "logic", "physics", "chemistry", "biology", "medicine", "history", "social sciences", "computer science", "business", "law", "philosophy"]

# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# pipe = pipeline("zero-shot-classification", model = model, tokenizer = tokenizer, device = "cuda")


# my_list = [99840, 99841]
# for i in my_list:
#     text = train.at[i, "question"]
#     output = pipe(text, TARGET_CLASSES)
#     train.loc[i, "subject"] = output["labels"][0]
    
# train.to_csv("data/subjects/train.csv", index=False)






# train = pd.read_csv("data/subjects/train.csv")
# TARGET_CLASSES = ["mathematics", "logic", "physics", "chemistry", "biology", "medicine", "history", "social sciences", "computer science", "business", "law", "philosophy", "miscellaneous"]
# for target in TARGET_CLASSES:
#     data = train["question"][train["subject_2"] == target]
#     data = data.sample(7)
#     print(f"CLASS: {target} ----------------------------------------------------------------------------------")
#     for val in data:
#         print(val)
#         print()
#     print()