import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Import Data
df = pd.read_csv("student_spending.csv")   # ✅ Ensure file exists here
print("Dataset preview:")
print(df.head(5))

# Encode categorical target (gender) into numeric labels
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# Feature(s) and Target
X = df[['age']]                  # Feature
y = df['gender_encoded']         # Target (numeric now)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Prediction for new input
x_new = float(input("Enter age: "))
y_new = model.predict([[x_new]])[0]
print(f"Predicted gender: {le.inverse_transform([y_new])[0]}")