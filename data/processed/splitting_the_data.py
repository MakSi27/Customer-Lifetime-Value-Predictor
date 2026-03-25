print("RUNNING NEW FILE")
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("customer_details_processed.csv")

# Step 2: Split into train (80%) and test (20%)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Step 3: Save to CSV files
train_df.to_csv("preprocessed_train.csv", index=False)
test_df.to_csv("preprocessed_test.csv", index=False)

print("Files created successfully!")