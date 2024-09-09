import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
df = pd.read_pickle('IMDB_BOW.pkl')
print("File read successfully.")
data = df.to_numpy()
np.random.shuffle(data)

# Scale features
scaler = StandardScaler()
x = scaler.fit_transform(data[:, 1:])
test_size = int(data.shape[0] / 10)

# Split data
x_train = x[2*test_size:]
x_val = x[test_size:2*test_size]
x_test = x[:test_size]
y_train = data[2*test_size:, 0]
y_val = data[test_size:2*test_size, 0]
y_test = data[:test_size, 0]

# Initialize models
models = {
     'Logistic Regression': LogisticRegression(max_iter=500),
    'SVM': SVC(kernel='linear', C=1, tol= 0.1),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Starting training for {name}")
    model.fit(x_train, y_train)
    print(f"Training complete for {name}")

    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# Print all results
for model, metrics in results.items():
    print(f"Results for {model}:")
    print(f"Accuracy: {metrics['Accuracy']:.2f}")
    print(f"Precision: {metrics['Precision']:.2f}")
    print(f"Recall: {metrics['Recall']:.2f}")
