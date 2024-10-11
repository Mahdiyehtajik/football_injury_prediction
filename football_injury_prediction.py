import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# To suppress any unnecessary warnings
import warnings
warnings.filterwarnings('ignore')
# Replace 'football_injury_data.csv' with the actual dataset file path
data = pd.read_csv('C:/Users/mahit/OneDrive/Desktop/football_injury_prediction/football_injury_data.csv')

# Check the structure of the dataset
data.head()
# Check for missing values
print(data.isnull().sum())

# If necessary, fill or drop missing values (example)
data.fillna(0, inplace=True)

# Convert categorical variables to numerical ones (e.g., position)
data['Position'] = data['Position'].astype('category').cat.codes

# Ensure that all data is numeric and clean
print(data.info())
# Example: Calculate the average minutes played per game
data['AvgMinutesPerGame'] = data['TotalMinutes'] / data['TotalGames']

np.random.seed(42)  
data['InjuryRisk'] = np.random.choice([0, 1], size=len(data), p=[0.7, 0.3])
data['GamesInLastMonth'] = np.random.randint(0, 5, size=len(data))  

# Calculate BMI
data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)

# Select features for modeling
features = ['Age', 'Position', 'AvgMinutesPerGame', 'GamesInLastMonth', 'PreviousInjuries', 'BMI']
X = data[features]
y = data['InjuryRisk']  # This is the target variable (1 = injured, 0 = not injured)
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC score (optional, but good for binary classification)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", roc_auc)
# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices])
plt.title('Feature Importances')
plt.show()
