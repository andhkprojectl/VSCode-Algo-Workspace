import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Read the CSV file
file_path = r'c:\tmp\1\markSix_20251005.csv'
df = pd.read_csv(file_path)

# Preprocess data
# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extract winning numbers as list
winning_cols = ['Winning Number 1', '2', '3', '4', '5', '6']
df['Winning Numbers'] = df[winning_cols].values.tolist()

# Create lagged previous winning numbers (shift 1 row)
df['Prev Winning Numbers'] = df['Winning Numbers'].shift(1)
df = df.dropna()  # Drop first row with no previous

# Create multi-label y: Binary indicators for 1-49 (1 if number is drawn, 0 otherwise)
numbers_range = range(1, 50)  # Mark Six is 1-49
y = pd.DataFrame(0, index=df.index, columns=numbers_range)
for i in range(len(df)):
    for num in df['Winning Numbers'].iloc[i]:
        y.loc[df.index[i], num] = 1

# Features: Other columns + previous winning numbers (flatten previous 6 numbers)
features = ['Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30', '31-40', '41-50']  # From the columns
prev_cols = [f'Prev Num {j}' for j in range(1, 7)]
df[prev_cols] = pd.DataFrame(df['Prev Winning Numbers'].tolist(), index=df.index)
X = df[features + prev_cols]

# Split data (use all but last for training, predict for next)
X_train, X_test, y_train, y_test = train_test_split(X[:-1], y[:-1], test_size=0.2, random_state=42)

# Train MultiOutputClassifier with RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
multi_rf = MultiOutputClassifier(rf, n_jobs=-1)
multi_rf.fit(X_train, y_train)

# Predict on the last row
last_row = X.iloc[-1:].copy()
#predicted_probs = np.array(multi_rf.predict_proba(last_row)[:, :, 1]).squeeze()  # Probabilities for each number being drawn

# last_row = np.random.rand(1, 5)
predicted_probs = np.array([p[0][1] for p in multi_rf.predict_proba(last_row)])

# top_6_indices = np.argsort(predicted_probs)[::-1][:6] + 1  # Top 6 numbers (1-based)
# predicted_numbers = sorted(top_6_indices)  # Sort for readability

top_8_indices = np.argsort(predicted_probs)[::-1][:8] + 1  # Top 6 numbers (1-based)
predicted_numbers = sorted(top_8_indices)  # Sort for readability

# Evaluate on test set (for demonstration)
y_pred = multi_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output
print("Model Accuracy on Test Set:", accuracy)
print("Predicted Next Winning 6 Numbers:", predicted_numbers)