import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# LOAD DATASET
df = pd.read_csv("pneumonia_dataset.csv")

print("✅ Dataset loaded successfully\n")
print(df.head())

# INPUT & OUTPUT
X = df.drop("pneumonia", axis=1)   
y = df["pneumonia"]                

# TRAIN TEST SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTotal rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

train_percent = (len(X_train)/len(df))*100
test_percent = (len(X_test)/len(df))*100

print("Training %:", train_percent)
print("Testing %:", test_percent)

# MODEL
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model trained successfully")

# PREDICTION
y_pred = model.predict(X_test)

# ACCURACY 
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", accuracy)

# CONFUSION MATRIX
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# CLASSIFICATION REPORT 
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# SAVE MODEL
joblib.dump(model, "pneumonia_model.pkl")
print("\n💾 Model saved as pneumonia_model.pkl")