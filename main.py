import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('TkAgg') # addresses a bug in macos: libc++abi.dylib: terminating with uncaught exception
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# style and color palette
sns.set(style="whitegrid")
current_palette = sns.color_palette()
# sns.palplot(current_palette)
# sns.palplot(sns.color_palette("husl", 8))
sns.set_palette("husl", 2)

# import dataset
df = pd.read_csv("assets/telecom-churn-small.csv")
# print(df.head(3))
# check if there is any null data
# print(df.isnull().values.any())

"""
# fill missing data
# df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')

# display current churn ratio -- TARGET VARIABLE
ax = sns.catplot(y="Churn", kind="count", data=df, height=2.6, aspect=2.5, orient='v')

#  numerical features
# TENURE is a customer's life (in months)
def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(df[df['Churn'] == 'No'][feature].dropna(), label= 'Churn: No')
    ax1 = sns.kdeplot(df[df['Churn'] == 'Yes'][feature].dropna(), label= 'Churn: Yes')
kdeplot('tenure')
kdeplot('MonthlyCharges')
kdeplot('TotalCharges')
"""

# categorical features
## binary features (Yes/No)
### senior citizen
def barplot_percentages(feature, orient='v', axis_name="percentage of customers"):
    ratios = pd.DataFrame()
    g = df.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name]/len(df)
    if orient == 'v':
        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()
barplot_percentages("SeniorCitizen")

## three unique values each

## our unique values


plt.figure(figsize=(12, 6))
df.drop(['customerID'], axis=1, inplace=True)
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap="YlGnBu")

plt.show()

