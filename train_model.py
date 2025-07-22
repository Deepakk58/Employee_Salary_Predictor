import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
df = pd.read_csv("adult_3.csv")  # Replace with actual filename

# 2. Remove rows with missing values or '?' entries
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# 3. Drop irrelevant or duplicate columns
df.drop(columns=['educational_num'], inplace=True)

# 4. Extract sample weights
sample_weights = df['fnlwgt']
df.drop(columns=['fnlwgt'], inplace=True)  # Drop from features

# 5. Split features and target
X = df.drop(columns=['income'])  # income is the label column
y = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

# 6. Encode categorical features
encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 7. Split data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42)

# 8. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train, sample_weight=w_train)

# 9. Save the model and encoders
joblib.dump(model, "predictor/ml/model.pkl")
joblib.dump(encoders, "predictor/ml/encoders.pkl")
