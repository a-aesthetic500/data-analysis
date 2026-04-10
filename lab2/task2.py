import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("StudentsPerformance 1.csv")

print(df.info())
print(df.head())

print(df.isnull().sum())

# Encode gender for correlation
df["gender_num"] = df["gender"].map({"male": 0, "female": 1})

# 3. Analyze relationships between variables
correlation = df.corr(numeric_only=True)
print(correlation)


# Visualization 1: Math score by gender
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="gender", y="math score")
plt.title("Math Score by Gender")
plt.show()

# Visualization 2: Reading vs Writing (correlation)
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="reading score",
    y="writing score"
)
plt.title("Relationship Between Reading and Writing Scores")
plt.show()

# Visualization 3: Score distribution by lunch type
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="lunch", y="math score")
plt.title("Math Score Distribution by Lunch Type")
plt.show()


# 1. Average math score by gender
avg_math_gender = df.groupby("gender")["math score"].mean()
print("Average math score by gender:")
print(avg_math_gender)

# 2. Correlation between reading and writing
print("Correlation between reading and writing:")
print(df["reading score"].corr(df["writing score"]))

# 3. Do test preparation courses help
prep_scores = df.groupby("test preparation course")[["math score", "reading score", "writing score"]].mean()
print("Test preparation course impact:")
print(prep_scores)

# 4. Best parental education level
edu_scores = df.groupby("parental level of education")[["math score", "reading score", "writing score"]].mean()
print("Scores by parental education level:")
print(edu_scores)

# 5. Score distribution by lunch type
lunch_scores = df.groupby("lunch")[["math score", "reading score", "writing score"]].mean()
print("Scores by lunch type:")
print(lunch_scores)
