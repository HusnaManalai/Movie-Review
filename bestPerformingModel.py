import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_pickle('IMDB_BOW.pkl')
print("Feature data loaded successfully.")

# Assuming sentiment labels are in 'IMDB_Dataset.csv' under a column named 'sentiment'
labels_df = pd.read_csv('IMDB_Dataset.csv')
y = labels_df['sentiment'].map({'positive': 1, 'negative': 0}).values  # Convert labels to binary
print("Labels loaded successfully.")

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(df.values, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Values of C to test
C_values = [0.000001, 10, 100000000]

# Dictionary to store results
results = {}

# Train and evaluate the model for each C value
for C in C_values:
    model = LogisticRegression(C=C, max_iter=500)
    model.fit(X_train_scaled, y_train)
    y_pred_val = model.predict(X_val_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val, zero_division=0)
    recall = recall_score(y_val, y_pred_val, zero_division=0)
    
    # Store results
    results[C] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# Output the results
for C, metrics in results.items():
    print(f"Results for C={C}:")
    print(f"  Accuracy: {metrics['Accuracy']:.2f}")
    print(f"  Precision: {metrics['Precision']:.2f}")
    print(f"  Recall: {metrics['Recall']:.2f}")
