import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load dataset

df = pd.read_csv("students.csv")


# Manual encoding


gender_map = {"Male":0, "Female":1, "Prefer not to say":2}
location_map = {"Rural (villages)":0, "Urban (Cities etc.)":1}
attendance_map = {">=90":2, ">=80":1, "<80":0}
sleep_map = {"2-4 hours":1, "4-6 hours":2, "6-8 hours":3, "More than 8 hours":4}
outings_map = {"No outings":0, "Once a week":1, "Twice a week":2, "3-5 times":3, "Everyday":4}
study_group_map = {"Yes":0, "No, i study alone.":1, "I study in group as well as alone.":2}
study_place_map = {"Home/Hostel Room":0, "Library/ Computer Centre":1, "Other":2}
notes_map = {
    "Yes, I mostly take notes in classes":2,
    "I sometimes take notes, like once or twice a week.":1,
    "No, I usually dont take notes in classes":0
}
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




# Features & Target

X = df.drop("previous_grade", axis=1)
y = df["previous_grade"]




# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




# Train Random Forest

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)



# Evaluate

y_pred = rf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# Visualize 


# Confusion matrix

cm = confusion_matrix(y_test, y_pred, labels=sorted(df['previous_grade'].unique()))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=sorted(df['previous_grade'].unique()), yticklabels=sorted(df['previous_grade'].unique()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.show()






# Qasim's Data

my_data = {
    "age": 20,              
    "gender": 0,            # 0=Male
    "hometown": 1,          # 1=Urban
    "study_time": 3,        
    "attendance": 2,        # 2 = >=90%
    "sleep": 3,             # 3 = 6-8 hours
    "outings": 1,           # 1 = Once a week
    "screen_time": 4,       
    "stress": 4,            
    "study_group": 2,       # 2 = Group & Alone 
    "study_place": 0,       # 0 = Home
    "notes": 2,             # 2 = Mostly take notes
    "food": 1,              # 1 = Mixed
    "physical_activity": 0  # 0 = Never 
}

#  CONVERT TO DATAFRAME
input_df = pd.DataFrame([my_data])


# PREDICT

prediction_probs = rf.predict_proba(input_df)
predicted_idx = np.argmax(prediction_probs, axis=1)[0]
confidence = prediction_probs[0][predicted_idx] * 100

# TRANSLATE RESULT
reverse_grade_map = {v: k for k, v in grade_map.items()}
final_grade = reverse_grade_map[predicted_idx]

print("\n" + "="*40)
print(f"RANDOM FOREST PREDICTION")
print("="*40)
print(f"Predicted Grade: {final_grade}")
print(f"Probability:     {confidence:.2f}%")
print("="*40 + "\n")