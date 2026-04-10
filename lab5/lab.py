import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
# Загрузка данных
df = pd.read_csv('StudentsPerformance 1.csv')

# 1. Проверка размера выборки (для ЦПТ)
n_total = len(df)
print(f"1. Размер выборки (N): {n_total}")
if n_total > 30:
    print("   Обоснование: Согласно ЦПТ, при N > 30 распределение выборочного среднего")
    print("   стремится к нормальному. Идеальная нормальность самих данных не обязательна.\n")

# 2. Проверка условия успеха/неудачи (Success/Failure Condition)
# Успех = Math Score > 80
female_scores = df[df['gender'] == 'female']['math score']
male_scores = df[df['gender'] == 'male']['math score']

f_success = (female_scores > 80).sum()
f_failure = (female_scores <= 80).sum()
m_success = (male_scores > 80).sum()
m_failure = (male_scores <= 80).sum()

print("2. Проверка условий для пропорций (Вопрос 4):")
print(f"   Женщины: Успехов = {f_success}, Неудач = {f_failure}")
print(f"   Мужчины: Успехов = {m_success}, Неудач = {m_failure}")

if all(x >= 10 for x in [f_success, f_failure, m_success, m_failure]):
    print("   Вердикт: Условие выполнено (все значения >= 10). Z-тест применим.")
else:
    print("   Вердикт: Условие НЕ выполнено. Нужно использовать точный тест Фишера.")


alpha = 0.05
print("----- QUESTION 1: One Sample T-Test -----")

t_stat, p1 = stats.ttest_1samp(df['math score'], 65)

print("p-value:", p1)

if p1 < alpha:
    print("Reject the Null Hypothesis")
else:
    print("Fail to Reject the Null Hypothesis")


print("\n----- QUESTION 2: Welch’s T-Test (One-Tailed) -----")

prep = df[df['test preparation course'] == 'completed']['math score']
none = df[df['test preparation course'] == 'none']['math score']

t_stat, p2_two = stats.ttest_ind(prep, none, equal_var=False)

# One-tailed correction
if prep.mean() > none.mean():
    p2 = p2_two / 2
else:
    p2 = 1 - (p2_two / 2)

print("p-value (one-tailed):", p2)

if p2 < alpha:
    print("Reject the Null Hypothesis")
else:
    print("Fail to Reject the Null Hypothesis")


print("\n----- QUESTION 3: Paired T-Test -----")

t_stat, p3 = stats.ttest_rel(df['reading score'], df['writing score'])

print("p-value:", p3)

if p3 < alpha:
    print("Reject the Null Hypothesis")
else:
    print("Fail to Reject the Null Hypothesis")


print("\n----- QUESTION 4: Proportion Z-Test -----")

# Success = math score > 80
df['success'] = df['math score'] > 80

female = df[df['gender'] == 'female']
male = df[df['gender'] == 'male']

success_female = female['success'].sum()
success_male = male['success'].sum()

n_female = len(female)
n_male = len(male)

count = [success_female, success_male]
nobs = [n_female, n_male]

stat, p4 = proportions_ztest(count, nobs)

print("p-value:", p4)

if p4 < alpha:
    print("Reject the Null Hypothesis")
else:
    print("Fail to Reject the Null Hypothesis")