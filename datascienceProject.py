import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


df = pd.read_csv(r"C:\Users\anmol\Downloads\Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States (1).csv")  

print("first five rows")
print(df.head())


print("\nLast five rows")
print(df.tail())

print("\nShape of the DataFrame")
print(df.shape)

print("\nrows, columns")
print(df.columns)

print("\nData types, non-null counts")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nData types of each column")
print(df.dtypes)

print("\nIndex Object")
print(df.index)

print("\nRandom 5 rows of the DataFrame")
print(df.sample(5))        

print("\nTotal missing values per column")
print(df.isnull().sum())

print("\nPercentage of missing values")
print(df.isnull().mean() * 100)

print("\nDropping rows which contains missing values in the column ESTIMATE")
df = df.dropna(subset=['ESTIMATE'])

print("\nAnalysing central tendency of death estimate agianst all persons")

all_persons_data = df[df['STUB_LABEL'] == 'All persons']['ESTIMATE']

clean_values = all_persons_data.dropna()

mean_val = clean_values.mean()
median_val = clean_values.median()
mode_val = clean_values.mode().iloc[0] if not clean_values.mode().empty else None

print(f"📈 Mean of death estimate for All persons: {mean_val}")
print(f"📊 Median of death estimate for All persons: {median_val}")
print(f"🎯 Mode of death estimate for All persons: {mode_val}")

#-------------------------z-test----------------------------------------------

print("\nperforming z-test on death estimate of different racial profile")
target_groups = [
    'Male: White',
    'Male: Black or African American',
    'Male: American Indian or Alaska Native',
    'Male: Asian or Pacific Islander',
    'Female: White',
    'Female: Black or African American',
    'Female: American Indian or Alaska Native',
    'Female: Asian or Pacific Islander'
]

# Filter relevant rows (just for 'All ages')
filtered_df = df[df['STUB_LABEL'].isin(target_groups) & (df['AGE'] == 'All ages')]

# Separate 2018 data and data from previous years
data_2018 = filtered_df[filtered_df['YEAR'] == 2018]['ESTIMATE']
data_prior = filtered_df[filtered_df['YEAR'] < 2018]['ESTIMATE']

# Calculate statistics
mean_2018 = data_2018.mean()
mean_prior = data_prior.mean()
std_prior = data_prior.std()
n_2018 = len(data_2018)

# Perform Z-test
z_score = (mean_2018 - mean_prior) / (std_prior / np.sqrt(n_2018))
p_value = 1 - norm.cdf(z_score)
# one-tailed test for "greater than"

# Print results
print(f"2018 Mean: {mean_2018:.2f}")
print(f"Prior Mean: {mean_prior:.2f}")
print(f"Z-score: {z_score:.3f}")
print(f"P-value: {p_value:.4f}")
print("Significant at 0.05 level:", p_value < 0.05)


#----------Line chart to visualise death estimate age wise over the years------------
#Clean Data

df_clean = df[df['ESTIMATE'].notnull() & df['AGE'].notnull() & df['YEAR'].notnull()]

#Style & Palette
# Optional: 'darkgrid', 'white', etc.
# Optional: 'deep', 'muted', 'pastel', etc.
sns.set_style('whitegrid')                      
sns.set_palette('coolwarm')                     


#Plotting Line Chart
# Set figure size for clarity
plt.figure(figsize=(12, 6))                     
line = sns.lineplot(data=df_clean,
                    x='YEAR',
                    y='ESTIMATE',
                    hue='AGE',
                    marker='o',                                 
                    linewidth=2.5,                              
                    palette='viridis'                           
)

#labels and title
plt.title("Death Estimate by Age Group Over the Years", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Death Rate (Estimate)", fontsize=12)

#Rotate x-axis ticks for readability
plt.xticks(rotation=45)

#Legend formatting
plt.legend(title="Age Group", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

#Final Layout Adjust
plt.tight_layout()
plt.grid(True)
plt.show()

#-----Bar Chart ------------------------------------

pd.set_option('display.max_rows', None)  # Show all rows

print(df[['STUB_LABEL', 'STUB_LABEL_NUM']].drop_duplicates().sort_values('STUB_LABEL_NUM'))

selected_codes = [4.1100, 4.1200, 4.1300, 4.1400, 4.2100, 4.2200, 4.2300, 4.2400]  # Add your exact codes here

race_selected = df[
    (df['STUB_LABEL_NUM'].isin(selected_codes)) &
    (df['ESTIMATE'].notnull())
]


plt.figure(figsize=(14, 6))
sns.barplot(data=race_selected, x='YEAR', y='ESTIMATE', hue='STUB_LABEL', dodge=True)
plt.title("Death Estimate by Year Race Groups")
plt.xlabel("Year")
plt.ylabel("Death Estimate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#-------box plot-----------------------------------------------------


# Filter only race groups (adjust STUB_LABEL manually if needed)
race_data = df[df['STUB_LABEL'].isin([
    'Male: White',
    'Male: Black or African American',
    'Male: American Indian or Alaska Native',
     'Male: Asian or Pacific Islander',
    'Female: White',
    'Female: Black or African American',
    'Female: American Indian or Alaska Native',
    'Female: Asian or Pacific Islander'
])]

plt.figure(figsize=(12, 6))
sns.boxplot(data=race_data, x='STUB_LABEL', y='ESTIMATE')
plt.title("Death Rate Distribution by Race Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#---------------------SCatter PLot-----------------------------------------


# Select necessary columns and drop missing values
df_clean = df[['YEAR', 'STUB_LABEL', 'ESTIMATE']].dropna()

# Filter only required STUB_LABELs
selected_labels = ['All persons', 'Male', 'Female']
filtered_df = df_clean[df_clean['STUB_LABEL'].isin(selected_labels)]

plt.figure(figsize=(12, 6))
# Scatter Plot
sns.scatterplot(data=filtered_df, x='YEAR', y='ESTIMATE', hue='STUB_LABEL', palette='viridis', s=100)

# Line Plot (connect same STUB_LABEL points)

sns.lineplot(data=filtered_df, x='YEAR', y='ESTIMATE', hue='STUB_LABEL', palette='viridis', legend=False)


plt.title('Death Rates Estimate Over Years for All Persons, Male & Female')
plt.xlabel('Year')
plt.ylabel('Death Rate (per 100,000)')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




#----------------HeatMap------------------------
# Clean fresh df for heatmap
df_heatmap = df[['YEAR', 'AGE', 'ESTIMATE']].dropna()

# Convert ESTIMATE to numeric
#df_heatmap['ESTIMATE'] = pd.to_numeric(df_heatmap['ESTIMATE'].astype(str).str.replace(',', ''), errors='coerce')

# Drop rows where ESTIMATE could not convert
df_heatmap = df_heatmap.dropna(subset=['ESTIMATE'])

# Create 5-year bins
df_heatmap['YEAR_GROUP'] = (df_heatmap['YEAR'] // 5) * 5

# Pivot Table: Average Death Rate in 5-year bins
heatmap_data = df_heatmap.pivot_table(
    values='ESTIMATE',
    index='AGE',
    columns='YEAR_GROUP',
    aggfunc='mean'
)

# Plot Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data, 
    annot=True,    # Values inside tiles
    fmt=".2f",     # 2 decimal points
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='grey'
)

plt.title('Death Rate Estimate by Age Group')
plt.xlabel('Year Group')
plt.ylabel('Age Group')
plt.tight_layout()
plt.show()











