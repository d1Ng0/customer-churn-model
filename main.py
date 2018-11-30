import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('TkAgg') # addresses a bug in macos: libc++abi.dylib: terminating with uncaught exception
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
df = pd.read_csv("assets/telecom-churn.csv")
# print(df.head(10))

# fill missing data
df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')
pd.options.display.max_rows = 500
pd.options.display.max_columns = 5000
pd.options.display.width = 100
print(df.head(10))


"""
#  KERNEL DENSITY ESTIMATION PLOT 
ax = sns.catplot(y="Churn", kind="count", data=df, height=2.6, aspect=2.5, orient='v')
def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(df[df['Churn'] == 'No'][feature].dropna(), label= 'Churn: No')
    ax1 = sns.kdeplot(df[df['Churn'] == 'Yes'][feature].dropna(), label= 'Churn: Yes')
kdeplot('tenure')
kdeplot('MonthlyCharges')
kdeplot('TotalCharges')
"""


"""
# categorical features
## binary features (Yes/No)
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
# barplot_percentages("SeniorCitizen")
# barplot_percentages("Partner")
# barplot_percentages("gender")
# barplot_percentages("Dependents")
# barplot_percentages("Contract")
"""

"""
# correlation table (pearson)
plt.figure(figsize=(12, 12))
df.drop(['customerID'], axis=1, inplace=True)
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap="YlGnBu")
ax.tick_params(labelsize=8)
"""

# ML classifier
params = {'random_state': 0, 'n_jobs': 4, 'n_estimators': 5000, 'max_depth': 8}
df = pd.get_dummies(df)
# Drop redundant columns (for features with two unique values)
drop = ['Churn_Yes', 'Churn_No', 'gender_Female', 'Partner_No', 'Dependents_No', 'PhoneService_No', 'PaperlessBilling_No']
x, y = df.drop(drop, axis = 1), df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Fit RandomForest Classifier
clf = RandomForestClassifier(**params)
clf = clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Model accuracy: {:.2f}".format(accuracy))
"""
"""

"""
# Plot features importances
imp = pd.Series(data = clf.feature_importances_, index = X_test.columns).sort_values(ascending = False)
imp = imp[imp>0.02]
# print(imp)
plt.figure(figsize=(10,12))
plt.title("Feature importance")
ax = sns.barplot(y=imp.index, x=imp.values, palette="husl", orient='h')
ax.tick_params(labelsize=8)
"""

plt.show()

# Use the forest's predict method on the test data
prediciton = clf.predict(X_test)
file_name = "assets/telecom-churn-prediction.csv"
prediction = pd.DataFrame(prediction, columns=['churn']).to_csv('prediction.csv')
# prediciton.to_csv(file_name, sep='\t', encoding='utf-8')
print("Done.")

"""
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
"""

