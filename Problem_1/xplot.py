import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("cleaned_merged_heart_dataset.csv")

# Heatmap for correlations
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
