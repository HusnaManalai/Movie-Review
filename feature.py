import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load data
features_df = pd.read_pickle('IMDB_BOW.pkl')
labels_df = pd.read_csv('IMDB_Dataset.csv')
print("Data loaded successfully.")

# Extract sentiment labels
labels = labels_df['sentiment'].map({'positive': 1, 'negative': 0}).values

# Add new textual features to the dataframe
# Assuming 'review' column exists in labels_df for textual content
labels_df['count_commas'] = labels_df['review'].apply(lambda text: text.count(','))
labels_df['count_periods'] = labels_df['review'].apply(lambda text: text.count('.'))
labels_df['count_all_caps'] = labels_df['review'].apply(lambda text: sum(word.isupper() for word in text.split()))

# Incorporate the new features into the original features dataframe
additional_features = labels_df[['count_commas', 'count_periods', 'count_all_caps']]
features_df = pd.concat([features_df, additional_features.reset_index(drop=True)], axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train and evaluate the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
precision = precision_score(y_val, y_pred_val, zero_division=0)
recall = recall_score(y_val, y_pred_val)

# Output results
print("Validation Results:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Evaluate the model on the test set
y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)
test_recall = recall_score(y_test, y_pred_test)
print("Test Results:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")