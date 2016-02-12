from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/train.csv', sep=';')

print(df.head())

print(df.dtypes)
