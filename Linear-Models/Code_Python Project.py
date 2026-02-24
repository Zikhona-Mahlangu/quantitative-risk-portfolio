#Python Project(Group_11)
from decimal import getcontext
getcontext().prec = 15

# ---- Imports ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro, ranksums

# Reading data
file_path = r"C:\STTN327\01_Project_Tyres_D.xlsx\01_Project_Tyres_D.xlsx"
all_sheets = pd.read_excel(file_path, sheet_name=None)

#Extract Treatments Sheet
Treatments = all_sheets['Treatments']

# PREPARE TREATED TYRE DATA
treated_drivers = Treatments[Treatments['treatment'] == 'y']['id'].tolist()
treated_data = pd.concat([all_sheets[driver] for driver in treated_drivers], ignore_index=False)

tread_columns = ['treadmeasureFL', 'treadmeasureFR', 'treadmeasureBL', 'treadmeasureBR']

# Interpolate missing tread measurements
for col in tread_columns:
    treated_data[col] = treated_data[col].interpolate(method='linear')

# Compute daily tread wear (negative difference)
for col in tread_columns:
    treated_data[f'{col}_diff'] = -(treated_data[col].diff())
    treated_data.loc[treated_data.index == 0, f'{col}_diff'] = 0

treated_data_clean = treated_data.dropna()

wheel_columns = ['FL', 'FR', 'BL', 'BR']
wheels_data = {}

for wheel in wheel_columns:
    wheel_df = treated_data_clean[[f'treadmeasure{wheel}_diff', 'distances', f'numshocks{wheel}']].copy()
    wheel_df.columns = ['treadmeasure_diff', 'distances', 'numshocks']
    wheel_df['treatment'] = 1
    wheels_data[wheel] = wheel_df

all_wheels = pd.concat(wheels_data.values(), ignore_index=True)

# PREPARE UNTREATED TYRE DATA
untreated_drivers = Treatments[Treatments['treatment'] == 'n']['id'].tolist()
untreated_data = pd.concat([all_sheets[driver] for driver in untreated_drivers], ignore_index=False)

for col in tread_columns:
    untreated_data[col] = untreated_data[col].interpolate(method='linear')

for col in tread_columns:
    untreated_data[f'{col}_diff'] = -(untreated_data[col].diff())
    untreated_data.loc[untreated_data.index == 0, f'{col}_diff'] = 0

untreated_data_clean = untreated_data.dropna()

untreated_wheels_data = {}

for wheel in wheel_columns:
    wheel_df = untreated_data_clean[[f'treadmeasure{wheel}_diff', 'distances', f'numshocks{wheel}']].copy()
    wheel_df.columns = ['treadmeasure_diff', 'distances', 'numshocks']
    wheel_df['treatment'] = 0
    untreated_wheels_data[wheel] = wheel_df

all_untreated_wheels = pd.concat(untreated_wheels_data.values(), ignore_index=True)

# Combine treated + untreated
all_wheels = pd.concat([all_wheels, all_untreated_wheels], ignore_index=True)

#LINEAR REGRESSION MODEL
X = all_wheels[['distances', 'numshocks', 'treatment']]
y = all_wheels['treadmeasure_diff']
X_with_const = sm.add_constant(X)

model = sm.OLS(y, X_with_const).fit()
robust_model = sm.OLS(y, X_with_const).fit(cov_type='HC3')
print(robust_model.summary())

#TREATMENT vs NO TREATMENT ANALYSIS
# Extract sheets manually for descriptive + nonparametric test
tyreData = all_sheets

# Combine treated data (ID1-ID15)
TreatmentTyres = pd.concat([tyreData[f'ID{i}'] for i in range(1, 16)], axis=0)
TreatmentTyres[tread_columns] = TreatmentTyres[tread_columns].interpolate(method="linear", limit_direction="both")
TreatmentTyres = TreatmentTyres.dropna()
TreatmentTyres['Treatment'] = 1
TreatmentTyres['Average'] = TreatmentTyres[tread_columns].mean(axis=1)

# Combine untreated data (ID16-ID30)
NoTreatmentTyres = pd.concat([tyreData[f'ID{i}'] for i in range(16, 31)], axis=0)
NoTreatmentTyres[tread_columns] = NoTreatmentTyres[tread_columns].interpolate(method="linear", limit_direction="both")
NoTreatmentTyres = NoTreatmentTyres.dropna()
NoTreatmentTyres['Treatment'] = 0
NoTreatmentTyres['Average'] = NoTreatmentTyres[tread_columns].mean(axis=1)

# Combined dataset
Data = pd.concat([NoTreatmentTyres, TreatmentTyres])
print(Data.head())

#NORMALITY TEST
sm.qqplot(Data["Average"], line="s")
plt.show()

# Shapiro-Wilk Test
tempM_new = (TreatmentTyres['Average']-np.mean(TreatmentTyres['Average']))/np.std(TreatmentTyres['Average'],ddof=1)
tempF_new = (NoTreatmentTyres['Average']-np.mean(NoTreatmentTyres['Average']))/np.std(NoTreatmentTyres['Average'],ddof=1)
poold = np.concatenate([tempM_new , tempF_new ])

shapiro(poold)

#NONPARAMETRIC TEST (Wilcoxon Rank-Sum)
ranksums(TreatmentTyres['Average'], NoTreatmentTyres['Average'], alternative="greater")


sns.violinplot(x="Treatment", y="Average", data=Data, palette="Blues", inner=None)
plt.title("Tread Depth Distribution: Treated vs Untreated Tyres")
plt.show()

#DRIVING HABIT ANALYSIS
DrivingHabits = pd.concat([TreatmentTyres, NoTreatmentTyres])
DrivingHabits['avg_shocks'] = DrivingHabits[['numshocksFL', 'numshocksFR', 'numshocksBL', 'numshocksBR']].mean(axis=1)

#Distance Distribution
plt.figure(figsize=(8, 5))
sns.histplot(DrivingHabits['distances'], bins=30, kde=True, color='teal')
plt.title("Distribution of Daily Driving Distances")
plt.xlabel("Distance Travelled (km)")
plt.ylabel("Frequency")
plt.show()

#Interpretation:
#According to the plot we observe that employees alot o f employees particularly drive between 0 - 100 km, 
#however theres fewer number of employees we can consider them as reckless drivers because they drive for longer than 100km in a day, which can be risky.


#Average Shock Distribution
plt.figure(figsize=(8, 5))
sns.histplot(DrivingHabits['avg_shocks'], bins=30, kde=True, color='orange')
plt.title("Distribution of Average Daily Shocks Experienced")
plt.xlabel("Average Number of Shocks per Day")
plt.ylabel("Frequency")
plt.show()

#Interpretation:
#The histogram shows that most employees drive smoothly on their daily basis, however there is a small number of employees which are less than 500 of them,
#which are reckless drivers since they experience and average shock of approximatley 0.5, which indicate that these employees they just hit the potholes without driving smooothly.


#Relationship between distance and shocks
plt.figure(figsize=(7, 5))
sns.scatterplot(x='distances', y='avg_shocks', data=DrivingHabits, alpha=0.6)
plt.title("Relationship Between Distance and Average Shocks")
plt.xlabel("Distance Travelled (km)")
plt.ylabel("Average Shocks per Day")
plt.show()

#Interpretation:
#The scatter plot shows no clear linear relationship between average shocks and distance,
#Which particularly indicate that employees who drive for a long distance doesn't experience more shocks.
#However, those who drive for short distance tend to drive less smoothly.


#Comparing driving distance between the groups
plt.figure(figsize=(7, 5))
sns.boxplot(x='Treatment', y='distances', data=DrivingHabits, palette="coolwarm")
plt.title("Comparison of Daily Driving Distance Between Groups")
plt.xlabel("Treatment (1 = Treated, 0 = Untreated)")
plt.ylabel("Distance Travelled (km)")
plt.show()

#The boxplot shows that employees with treated and untreated tyres drove similar daily distances on average.
#This suggests that driving exposure was evenly distributed across participants, supporting the fairness of the experimental design.

#Overall Conclusion:
#Drivers drove conservatively and under the same conditions, regardless of tyre treatment. 
#Driving distances and shock exposure per group were evenly distributed, 
#which indicates a reasonable and well-controlled study. 
#Therefore, it is safe to conclude that the improved tread durability in the treated tyres was a consequence of the effectiveness of the new tyre treatment rather than a function of variability in employee driving habits.


