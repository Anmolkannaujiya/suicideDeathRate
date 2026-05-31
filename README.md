Public Health Data Analysis — Python EDA & Statistical Testing
Overview
An end-to-end Exploratory Data Analysis (EDA) project on a real-world CDC government dataset analyzing mortality trends across demographic groups in the United States. The project combines data cleaning, statistical hypothesis testing, and multi-type visualizations to surface meaningful public health insights.

Objectives

Clean and prepare raw government data for reliable analysis
Analyze central tendency of death estimates across demographic groups
Statistically compare mortality rates across racial and gender groups using hypothesis testing
Visualize trends, distributions, and patterns across age, sex, race, and time
Identify outliers and anomalies in demographic mortality data


Dataset

Source: CDC / data.gov (US Government public dataset)
Name: Death Rates for Suicide by Sex, Race, Hispanic Origin, and Age — United States
Key Fields: Year, Age Group, Sex, Race/Ethnicity, Death Rate Estimate


Tools & Libraries
LibraryUsagePandasData loading, cleaning, manipulationNumPyNumerical operations and statistical calculationsSeabornStatistical visualizationsMatplotlibPlot customization and layoutSciPy (stats)Z-test hypothesis testing

Project Workflow
1. Data Exploration

Loaded and inspected dataset using .head(), .tail(), .shape(), .info(), .describe()
Identified missing values per column and calculated missing percentage
Dropped rows with null values in the ESTIMATE column

2. Central Tendency Analysis

Filtered data for "All persons" demographic group
Calculated mean, median, and mode of death rate estimates

3. Statistical Hypothesis Testing — Z-Test

Hypothesis: Did 2018 death rates differ significantly from prior years across racial and gender groups?
Filtered data for 8 racial/gender groups at "All ages" level
Compared 2018 mean vs prior years mean using a one-tailed Z-test
Result: Determined statistical significance at 0.05 confidence level

4. Visualizations
ChartPurposeLine ChartDeath rate trends by age group over yearsBar ChartYear-on-year comparison across racial groupsBox PlotDistribution and outlier detection by race/genderScatter + Line PlotTrend comparison for All persons, Male, FemaleHeatmapAverage death rate by age group across 5-year time bins

Key Insights

Identified age groups with consistently higher mortality rates over time
Detected statistically significant differences in 2018 death rates vs prior years across racial groups
Box plots revealed considerable outliers in specific demographic segments
Heatmap showed clear clustering of elevated rates in older age groups across specific time periods


Skills Demonstrated

Data cleaning and missing value treatment
Descriptive statistical analysis
Hypothesis testing (Z-test) using SciPy
Multi-type data visualization using Seaborn and Matplotlib
Demographic trend analysis and insight generation
Working with real-world government datasets
