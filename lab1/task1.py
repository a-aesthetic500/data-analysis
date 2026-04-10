import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("StudentsPerformance 1.csv")
df.info()
df["gender_num"]=df["gender"].map({"male":0,"female":1})
cor = df.corr(numeric_only=True)
print(cor)
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="gender_num",
    y="math score",
)
plt.title("relation between gender and math score")
plt.xlabel("gender")
plt.ylabel("math score")
plt.grid(True)
plt.show()

# Correlation analysis shows a strong positive relationship between
# math, reading, and writing scores (0.80–0.95).
# The strongest correlation is observed between reading and writing scores.
# Gender has a weak relationship with academic performance:
# females slightly perform better in reading and writing,
# while math scores are slightly lower on average.
