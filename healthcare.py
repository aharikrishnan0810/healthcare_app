# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For saving the model
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler  # For scaling features (optional, good for consistency)

# Load dataset
# Assuming the dataset is in a CSV file called 'diabetes.csv'
df = pd.read_csv('diabetes.csv')

# Handling missing values (if any) - replacing them with the mean of the column
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separate features and target variable
X = df_imputed.drop(columns=['Outcome'])  # Features (Pregnancies, Glucose, etc.)
y = df_imputed['Outcome']  # Target (1 for Diabetes, 0 for No Diabetes)

# Feature Scaling (optional but useful for many models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier with hyperparameters
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'random_forest_diabetes_model.pkl'
joblib.dump(rf_classifier, model_filename)

print(f"Model saved as {model_filename}")

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Optionally save the scaler (if scaling is applied) to use during inference
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as {scaler_filename}")
