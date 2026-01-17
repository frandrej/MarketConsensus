import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your training data
# X_train should be a DataFrame with the same features used in training
X_train = pd.read_csv(r'C:\Users\diefr\OneDrive\Lernsachen\MarketConsensus\data\processed\features_dataset.csv')

# Get feature names from feature_importance_fixed.csv
feature_importance = pd.read_csv(r'C:\Users\diefr\OneDrive\Lernsachen\MarketConsensus\data\processed\feature_importance_fixed.csv')
feature_names = feature_importance['feature'].tolist()
X_train = X_train[feature_names]

# 1. Create imputation dict (median of each feature)
imputation_dict = {}
for col in feature_names:
    median_val = X_train[col].median()
    imputation_dict[col] = median_val if not pd.isna(median_val) else 0

# 2. Apply imputation
for col in feature_names:
    X_train[col].fillna(imputation_dict[col], inplace=True)

# 3. Create and fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# 4. Save both
with open('scaler_fixed.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('imputation_dict_fixed.pkl', 'wb') as f:
    pickle.dump(imputation_dict, f)

print("✓ Created scaler_fixed.pkl and imputation_dict_fixed.pkl")