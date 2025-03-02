import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing data and creating dataframe ----------------------------------------!
tb = pd.read_csv('/Users/sissizhou/Downloads/SDSS Datathon Cases/Transportation/subway-data-cleaned.csv')
df = pd.DataFrame(tb)

#Plots ----------------------------------------------------------------------!

df_copy = df[df['Min Delay'] != 0].copy()

#Peak time for delays--------------------------------------------------------
#For all days
df_all = df_copy.groupby('Time')['Min Delay'].mean().reset_index()

all = df_all.plot('Time', 'Min Delay', linestyle='-')

all.set_title('Peak Times for Delays')
all.set_xlabel('Time')
all.set_ylabel('Average Delay (minutes)')

#Geographical Hotspots -----------------------------------------------------!
#Classify delays (in minutes) into 3 categories: mild (0-2), moderate (2-10), severe(>10)
#Create a new column 'Degree' in df_classified
#Count the number of delays in each category for each station
#Plot a stacked bar chart of the number of delays in each category for each station
df_classified = df.copy()

df_classified['Degree'] = df_classified['Min Delay'].apply(
    lambda x: 'Severe' if x > 10 else ('Moderate' if x >= 2 else 'None'))

df_classified = df_classified.groupby(['Station', 'Degree']).size().reset_index(name='Count')

df_classified_severe = df_classified[df_classified['Degree'] == 'Severe']
df_classified_moderate = df_classified[df_classified['Degree'] == 'Moderate']
df_classified_none = df_classified[df_classified['Degree'] == 'None']

df_combined = pd.concat([df_classified_none, df_classified_moderate, df_classified_severe], ignore_index=True)

df_combined = df_combined.pivot(index='Station', columns='Degree', values='Count')

df_combined.plot(kind='bar', stacked=True, figsize=(8, 6), color=['green', 'yellow', 'red'])

plt.xlabel('Station')
plt.ylabel('Number of Delays by Severity')
plt.title('Number of Delays by Severity at Each Station')
plt.legend(title='Severity')

plt.tight_layout()
plt.show()