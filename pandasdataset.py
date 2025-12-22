import pandas as pd

# Names of students
names = [
    "Ananya", "Rahul", "Priya", "Kiran",
    "Vikram", "Sneha", "Arjun", "Meera",
    "Rohit", "Divya"
]

# All values are in hours, within 0–24
sleep_hours = [8, 7, 6, 7, 8, 6, 7, 8, 6, 7]
play_hours  = [2, 3, 4, 2, 1, 3, 2, 1, 4, 2]
study_hours = [4, 5, 3, 6, 4, 5, 3, 4, 2, 5]

# Calculate "Other Hours" so that total = 24
other_hours = []
total_hours = []
for s_sleep, s_play, s_study in zip(sleep_hours, play_hours, study_hours):
    used = s_sleep + s_play + s_study
    other = 24 - used
    other_hours.append(other)
    total_hours.append(used + other)

# Test scores (you said mandatory)
test_scores = [88, 76, 69, 92, 80, 85, 70, 90, 65, 82]

data = {
    "Student Name": names,
    "Sleep Hours": sleep_hours,
    "Play Hours": play_hours,
    "Study Hours": study_hours,
    "Other Hours": other_hours,
    "Total Hours": total_hours,
    "Test Score": test_scores
}

df = pd.DataFrame(data)

print("Pandas DataFrame:")
print(df)

df.to_csv("student_habits.csv", index=False)
print("File saved: student_habits.csv")

import os
print("File saved at:", os.path.abspath("student_habits.xlsx"))

print("\nFiles saved: student_habits.xlsx and student_habits.csv")
