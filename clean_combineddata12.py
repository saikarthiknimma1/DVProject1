import pandas as pd

# Load the original dataset
df12 = pd.read_csv('static/data/CombinedData12.csv')

# Rename columns for consistency
df12.rename(columns={
    'publish_date': 'date',
    'Text': 'text',
    'Sentiment': 'sentiment',
    'Source': 'source'
}, inplace=True)

# Convert 'date' from YYYYMMDD to datetime
df12['date'] = pd.to_datetime(df12['date'], format='%Y%m%d', errors='coerce')

# Drop rows with missing values in key columns
df12.dropna(subset=['date', 'text', 'sentiment'], inplace=True)

# Standardize text
df12['sentiment'] = df12['sentiment'].str.strip().str.lower()
df12['source'] = df12['source'].str.strip().str.lower()

# Filter only valid sentiments
valid_sentiments = ['positive', 'neutral', 'negative']
df12 = df12[df12['sentiment'].isin(valid_sentiments)]

# Drop duplicates
df12.drop_duplicates(inplace=True)

# Limit to 20,003 rows
df12 = df12.head(20003)

# Save cleaned dataset
df12.to_csv('static/data/Cleaned_CombinedData12.csv', index=False)

# Preview
print(df12.head())
