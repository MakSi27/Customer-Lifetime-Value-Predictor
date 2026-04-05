import pandas as pd

#1. Basic preprocessing (optional)

def preprocess_data(df):
    df = df.drop_duplicates()
    return df

# 2. Create LTV (TARGET VARIABLE)

def create_ltv(df):
    df['ltv'] = (0.30 * df['Purchase Amount (USD)'] + 0.25 * df['Frequency of Purchases'] + 0.20 * df['Previous Purchases'] + 0.15 * df['Review Rating'] + 0.10 * df['Subscription Status'])
    return df

# 3. Full processing pipeline

def process_pipeline(df):
    df = preprocess_data(df)
    df = create_ltv(df)

    return df