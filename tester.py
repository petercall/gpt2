import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np





# train = pd.read_csv("data/subjects/train.csv")

# MODEL_NAME = "facebook/bart-large-mnli"

# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# pipe = pipeline("zero-shot-classification", model = model, tokenizer = tokenizer, device = "cuda")


# my_list = [99840, 99841]
# for i in my_list:
#     text = train.at[i, "question"]
#     output = pipe(text, TARGET_CLASSES)
#     train.loc[i, "subject_2"] = output["labels"][0]
    
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