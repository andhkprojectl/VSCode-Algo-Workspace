import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Target DIA and YM close price
def performCorrelation(df1):
    df2 = df1.copy()
    # remove string column or not related column before run correlatino
    # df2 = df2.drop(columns=['Date1_1', 'Time1_1', 'symbol_1', 'Date1_2', 'Time1_2', 'symbol_2'])

    # Save to new CSV file
    df2.to_csv('c:/tmp/performCorrelationNQ.csv', index=False)

    corr_matrix = df2.corr()
    # display
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Prevent line wrapping
    pd.set_option('display.max_colwidth', None)  # Full column text
    # print ("rt_1")
    #print (corr_matrix["rt_1"].sort_values(ascending=False))
    # print("")
    df2_rt2 = corr_matrix["maxRtPstL0"].sort_values(ascending=False)
    # df2_rt2 = corr_matrix["ptl_top_2_p"].sort_values(ascending=False)
    # print("rt_2")
    print (df2_rt2)
    df2_rt2.to_frame(name="Correlation").to_csv("c:/tmp/performCorrelation10.csv", index_label="Field")

# Load data
# df = pd.read_csv('c:/tmp/aaa11_negative.csv')
df = pd.read_csv('c:/tmp/aaa2.csv')

# Drop non-numeric column 'Date1'
df = df.drop(columns=['Date1'])
# df = df.drop(columns=['Time1'])
df = df.drop(columns=['Time1'], errors='ignore')
df = df.drop(columns=['maxRtPst'], errors='ignore')


# correlation
performCorrelation(df)

# Create combinations
# df['comb_pos'] = df['sRank3'] + df['rsi14']  # Positive combination for maxRtPst = 1
# df['comb_neg'] = df['sRank4'] + df['sTotalBar']  # Negative combination for maxRtPst = 0


# optional drop some columns Coefficient is low (at beginning. enable all columns, check which coeff has log coeff)
# remove all columns abs(coeff) < 0.1

# df = df.drop(columns=['RegularHourTotalV'])
# df = df.drop(columns=['rsi14'])
# df = df.drop(columns=['sRank5%'])
# df = df.drop(columns=['VWithPst'])
# df = df.drop(columns=['sTotalBar'])
# df = df.drop(columns=['sRank3%'])


# Features (X): Original + combinations
X = df.drop(columns=['maxRtPstL0'])
y = df['maxRtPstL0']


# Handle NaN values
X = X.fillna(0)
y = y.fillna(0)


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)



# Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Coefficients for interpretation
features = X.columns
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0],
    'Abs Coefficient': np.abs(model.coef_[0]),
    'Odds Ratio': np.exp(model.coef_[0])
}).sort_values('Abs Coefficient', ascending=False)



print("Feature Importance (Coefficients):")
print(coef_df)

# Evaluate
y_pred = model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))

# Predict on last row (as "new" data)
last_row = X.iloc[-1:].values
last_row_scaled = scaler.transform(last_row)
prob = model.predict_proba(last_row_scaled)[0, 1]
predicted_y = model.predict(last_row_scaled)[0]

print("\nPrediction for Last Row (maxRtPstL0 = 1 probability):", prob)
print("Predicted maxRtPstL0:", predicted_y)

