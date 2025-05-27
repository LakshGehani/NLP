import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load prediction data ---
# Update path if needed
df = pd.read_csv("data/sentiment_news_PLTR_finetuned_finbert.csv")

# --- Sentiment Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="finbert_label", order=[
              "positive", "neutral", "negative"])
plt.title("PLTR News Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Headlines")
plt.tight_layout()
os.makedirs("images", exist_ok=True)
plt.savefig("plots/PLTR_sentiment_distribution.png")
plt.close()

