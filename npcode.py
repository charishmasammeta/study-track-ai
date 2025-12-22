import numpy as np

# -----------------------------------
# 1️⃣ CREATE NUMPY ARRAYS
# -----------------------------------
study_hours = np.array([4, 6, 3, 5, 2, 7, 1, 8])
sleep_hours = np.array([8, 7, 6, 7, 8, 6, 7, 5])
test_scores = np.array([88, 76, 69, 92, 80, 85, 70, 90])

print("Study Hours Array:")
print(study_hours)

print("\nSleep Hours Array:")
print(sleep_hours)

print("\nTest Scores Array:")
print(test_scores)

# -----------------------------------
# 2️⃣ BASIC NUMPY MATHEMATICAL OPERATIONS
# -----------------------------------

# Addition
total_hours = study_hours + sleep_hours
print("\nTotal Hours (Study + Sleep):")
print(total_hours)

# Subtraction
difference = sleep_hours - study_hours
print("\nDifference (Sleep - Study):")
print(difference)

# Multiplication
multiply = study_hours * 2
print("\nStudy Hours × 2:")
print(multiply)

# Division
division = sleep_hours / 2
print("\nSleep Hours ÷ 2:")
print(division)

# -----------------------------------
# 3️⃣ ARRAY STATISTICS
# -----------------------------------

print("\n--- ARRAY STATISTICS ---")
print("Maximum Study Hours:", np.max(study_hours))
print("Minimum Study Hours:", np.min(study_hours))
print("Average Study Hours:", np.mean(study_hours))
print("Median Study Hours :", np.median(study_hours))
print("Standard Deviation:", np.std(study_hours))

# -----------------------------------
# 4️⃣ ADVANCED NUMPY OPERATIONS
# -----------------------------------

# Dot product → Measures relationship between 2 arrays
dot_product = np.dot(study_hours, test_scores)
print("\nDot Product (Study × Score):", dot_product)

# Element-wise square
study_square = np.square(study_hours)
print("\nSquare of Study Hours:")
print(study_square)

# Conditional filtering
high_scores = test_scores[test_scores > 80]
print("\nScores Above 80:")
print(high_scores)
