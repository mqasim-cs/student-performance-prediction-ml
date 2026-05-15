import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#Load Data
df = pd.read_csv("students.csv")

# Manual Encoding
gender_map = {"Male":0, "Female":1, "Prefer not to say":2}
location_map = {"Rural (villages)":0, "Urban (Cities etc.)":1}
attendance_map = {">=90":2, ">=80":1, "<80":0}
sleep_map = {"2-4 hours":1, "4-6 hours":2, "6-8 hours":3, "More than 8 hours":4}
outings_map = {"No outings":0, "Once a week":1, "Twice a week":2, "3-5 times":3, "Everyday":4}
study_group_map = {"Yes":0, "No, i study alone.":1, "I study in group as well as alone.":2}
study_place_map = {"Home/Hostel Room":0, "Library/ Computer Centre":1, "Other":2}
notes_map = {"Yes, I mostly take notes in classes":2, "I sometimes take notes, like once or twice a week.":1, "No, I usually dont take notes in classes":0}
food_map = {"Healthy":2, "Mixed":1, "Junk":0}
physical_map = {"Never":0, "1-2 times":1, "3-5 times":2, "Daily":3}
grade_map = {"4":5, "3.67 - 3.99":4, "3.33 - 3.66":3, "3.0 - 3.32":2, "2.67 - 2.99":1, "<2.67":0}

df["gender"] = df["gender"].map(gender_map)
df["hometown"] = df["hometown"].map(location_map)
df["attendance"] = df["attendance"].map(attendance_map)
df["sleep"] = df["sleep"].map(sleep_map)
df["outings"] = df["outings"].map(outings_map)
df["study_group"] = df["study_group"].map(study_group_map)
df["study_place"] = df["study_place"].map(study_place_map)
df["notes"] = df["notes"].map(notes_map)
df["food"] = df["food"].map(food_map)
df["physical_activity"] = df["physical_activity"].map(physical_map)
df["previous_grade"] = df["previous_grade"].map(grade_map)

df.dropna(inplace=True)

# Features & Target
X = df[["age", "gender", "hometown", "study_time", "attendance", "sleep", "outings", 
        "screen_time", "stress", "study_group", "study_place", "notes", "food", "physical_activity"]]
y = df["previous_grade"]

# Train test Split and Scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build  Neural Network

model = Sequential([

    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')

])

# Compile & Train

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_scaled, y_train, 
    epochs=150, 
    batch_size=32, 
    validation_data=(X_test_scaled, y_test),
    verbose=1
)


# Evaluate

# Get probabilities
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Visualizations

# Loss Curve 
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title("Model Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(df['previous_grade'].unique()))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=sorted(df['previous_grade'].unique()), yticklabels=sorted(df['previous_grade'].unique()))
plt.title("Confusion Matrix (Neural Netowrk)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.show()

my_data = {
    "age": 20,              # my age
    "gender": 0,            # 0=Male, 1=Female
    "hometown": 1,          # 0=Rural, 1=Urban
    "study_time": 3,        
    "attendance": 2,        # 2 = >=90%
    "sleep": 3,             # 3 = 6-8 hours
    "outings": 1,           # 1 = Once a week
    "screen_time": 4,       
    "stress": 4,            
    "study_group": 2,       # 1 = Study alone
    "study_place": 0,       # 1 = Library
    "notes": 2,             # 2 = Take notes always
    "food": 1,              # 2 = Healthy
    "physical_activity": 0  # 3 = Daily
}

# CONVERT TO ARRAY

input_data = np.array([list(my_data.values())])

# SCALE THE DATA
input_scaled = scaler.transform(input_data)

# PREDICT

prediction_prob = model.predict(input_scaled)

predicted_class_index = np.argmax(prediction_prob)

# TRANSLATE BACK TO TEXT

reverse_grade_map = {v: k for k, v in grade_map.items()}
predicted_gpa_text = reverse_grade_map[predicted_class_index]
confidence_score = prediction_prob[0][predicted_class_index] * 100

print("\n" + "="*40)
print(f"PREDICTION RESULTS")
print("="*40)
print(f"Predicted Grade Category: {predicted_gpa_text}")
print(f"Confidence Level:         {confidence_score:.2f}%")
print("="*40 + "\n")