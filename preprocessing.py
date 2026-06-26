import pandas as pd
import re

df = pd.read_csv("dataset/IMDB Dataset.csv")

# Convert to lowercase
df['review'] = df['review'].str.lower()

# Remove HTML tags
df['review'] = df['review'].apply(
    lambda x: re.sub(r'<.*?>', '', x)
)

# Remove punctuation
df['review'] = df['review'].apply(
    lambda x: re.sub(r'[^\w\s]', '', x)
)

df['review'] = df['review'].apply(
    lambda x: re.sub(r'\s+', ' ', x).strip()
)

print(df['review'][0])
df.to_csv("dataset/cleaned_imdb.csv", index=False)

print("Cleaned dataset saved successfully!")