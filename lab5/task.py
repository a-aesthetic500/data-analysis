import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# =========================
# Load dataset
# =========================
df = pd.read_csv("StudentsPerformance 1.csv")

print("Total sample size:", len(df))
print("\n")

# =========================
# PART 1 — Assumptions
# =========================

# 1️⃣ Normality (CLT)
print("Normality Check:")
print("Since N =", len(df), "> 30, by Central Limit Theorem normality is not required.\n")

# 2️⃣ Success/Failure Condition
df["Success"] = df["math score"] > 80

print("Success/Failure Counts by Gender:")
print(df.groupby("gender")["Success"].value_counts())
print("\n")

# =========================
# PART 2 — Hypothesis Tests
# =========================

# =========================
# Question 1
# One-sample t-test
# =========================
print("Question 1: One-sample t-test vs 65")

t1, p1 = stats.ttest_1samp(df["math score"], 65)

print("t-stat =", t1)
print("p-value =", p1)

if p1 < 0.05:
    print("Decision: Reject H0\n")
else:
    print("Decision: Fail to Reject H0\n")

# =========================
# Question 2
# Welch's T-test (one-tailed)
# =========================
print("Question 2: Welch’s T-test (Prep vs None)")

prep = df[df["test preparation course"] == "completed"]["math score"]
none = df[df["test preparation course"] == "none"]["math score"]

t2, p2_two = stats.ttest_ind(prep, none, equal_var=False)

# one-tailed
p2 = p2_two / 2

print("t-stat =", t2)
print("one-tailed p-value =", p2)

if p2 < 0.05 and t2 > 0:
    print("Decision: Reject H0\n")
else:
    print("Decision: Fail to Reject H0\n")

# =========================
# Question 3
# Paired t-test
# =========================
print("Question 3: Paired T-test (Reading vs Writing)")

t3, p3 = stats.ttest_rel(df["reading score"], df["writing score"])

print("t-stat =", t3)
print("p-value =", p3)

if p3 < 0.05:
    print("Decision: Reject H0\n")
else:
    print("Decision: Fail to Reject H0\n")

# =========================
# Question 4
# Proportion Z-test
# =========================
print("Question 4: Proportion Z-test (Gender Excellence)")

female = df[df["gender"] == "female"]
male = df[df["gender"] == "male"]

x1 = sum(female["math score"] > 80)
n1 = len(female)

x2 = sum(male["math score"] > 80)
n2 = len(male)

print("Female Success:", x1, "/", n1)
print("Male Success:", x2, "/", n2)

# pooled proportion
p_pool = (x1 + x2) / (n1 + n2)
print("Pooled proportion =", p_pool)

count = np.array([x1, x2])
nobs = np.array([n1, n2])

z4, p4 = proportions_ztest(count, nobs)

print("z-stat =", z4)
print("p-value =", p4)

if p4 < 0.05:
    print("Decision: Reject H0")
else:
    print("Decision: Fail to Reject H0")