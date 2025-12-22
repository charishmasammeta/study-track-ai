import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file (student dataset you generated earlier)
df = pd.read_csv("student_habits.csv")

# --------------------------------------------------
# 1️⃣ Seaborn - Stylish Scatter Plot
# --------------------------------------------------
plt.figure(figsize=(7, 4))
sns.scatterplot(
    x="Study Hours",
    y="Test Score",
    data=df,
    hue="Student Name"
)
plt.title("Study Hours vs Test Score")
plt.show()

# --------------------------------------------------
# 2️⃣ Seaborn - Stylish Bar Plot
# --------------------------------------------------
plt.figure(figsize=(7, 4))
sns.barplot(x="Student Name", y="Sleep Hours", data=df)
plt.title("Sleep Hours per Student")
plt.xticks(rotation=45)
plt.show()

# --------------------------------------------------
# 3️⃣ Matplotlib - Line Chart
# --------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(df["Student Name"], df["Study Hours"], marker="o")
plt.title("Study Hours Trend")
plt.xlabel("Student")
plt.ylabel("Study Hours")
plt.xticks(rotation=45)
plt.show()

# --------------------------------------------------
# 4️⃣ Seaborn - Heatmap (Stylish Correlation Chart)
# --------------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

