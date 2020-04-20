import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, lars_path
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Grab our fights
with open("pickles/merged.pickle", 'rb') as to_read:
    merged = pickle.load(to_read)

merged['male'] = pd.get_dummies(merged['male'], drop_first=True)

# Let's not bet on tomato cans
merged = merged[merged['fight_count_sum'] < 100]

###########################################
## First Plot: Distribution of Fight Counts
###########################################
fight_counts_agg = df\
    .groupby(['f1'])\
    .agg({
        'fight_count': 'first',
    })
fight_counts_agg.hist()

# Our dataset clearly skews towards fighters with very few fights

# Let's make it more balanced by fight counts, and gender too, why not
men = merged[merged['male'] == 1]
women = merged[merged['male'] == 0]

quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
quantile_labels = quantiles[1:]

for quantile in quantiles:
    print(f"{quantile:.2f}")
    print(f"men: \t{men['fight_count_sum'].quantile(quantile)}")
    print(f"women: \t{women['fight_count_sum'].quantile(quantile)}")

men['fights_quantile'] = pd.qcut(men['fight_count_sum'],
        q=4, precision=0, labels=quantile_labels)
women['fights_quantile'] = pd.qcut(women['fight_count_sum'],
        q=4, precision=0, labels=quantile_labels)

men_by_quantile = {}
women_by_quantile = {}
for quantile in quantiles[1:]:
    men_by_quantile[quantile] = \
            men[men['fights_quantile'] == quantile].iloc[:1_000]
    women_by_quantile[quantile] = \
            women[women['fights_quantile'] == quantile].iloc[:1_000]

balanced_men = pd.concat([df for quantile, df in men_by_quantile.items()])
balanced_women = pd.concat([df for quantile, df in women_by_quantile.items()])

balanced = pd.concat([balanced_men, balanced_women])

###########################################
## EDA (a little of it)
###########################################

# There were too many features at this point to easily look at the pairplot
# (Or even to quickly generate one)
# So I looked at them in groups
# The currently-uncommented group shows interesting correlations with height

# cols = ['fight_count', 'height', 'class', 'male', 'fight_count_2', 'time']
cols = ['height_2', 'fight_count_dif', 'fight_count_sum', 'height_dif', 'height_sum', 'time']
# cols = ['win_rate_dif', 'win_rate_sum', 'loss_rate_dif', 'loss_rate_sum', 'fastest_win_dif', 'time']
# cols = ['fastest_win_sum', 'fastest_loss_dif', 'fastest_loss_sum', 'avg_win_time_dif', 'time']
# cols = ['avg_win_time_sum', 'avg_loss_time_dif', 'avg_loss_time_sum', 'fights_quantile', 'time']
sns.pairplot(balanced[cols])

###########################################
## Regression
###########################################

X = balanced.drop(['time'], 1)
y = balanced['time']

# I played around with different feature inputs
# sumcols = [col for col in balanced.columns if 'sum' in col]
# difcols = [col for col in balanced.columns if 'dif' in col]
# Xsum = balanced.drop(difcols + ['time'], 1)
# Xdif = balanced.drop(sumcols + ['time'], 1)
# X = Xsum
# X = Xdif

# I played around with other things too, but the code was way too messy

# Anyway...train/test/val:
X_train_val, X_test, y_train_val, y_test = \
        train_test_split(X, y, test_size=0.2,random_state=42)
X_train, X_val, y_train, y_val = \
        train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

# Scale the data
std = StandardScaler()
std.fit(X_train.values)
X_tr = std.transform(X_train.values)
X_te = std.transform(X_test.values)

# Cross-validate and re-fit
alphavec = 10 ** np.linspace(-2, 2, 200)
lasso_model = LassoCV(alphas=alphavec, cv=5)
lasso_model.fit(X_tr, y_train)

# Coefficients
list(zip(X_train.columns, lasso_model.coef_))

# Mean Absolute Error, copied from notes
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

# Predict and score
test_set_pred = lasso_model.predict(X_te)
mae(y_test, test_set_pred)
# 4 min 26 seconds wrong, on average
r2_score(y_test, test_set_pred)

###########################################
## LARS Path
###########################################

alphas, _, coefs = lars_path(X_tr, y_train.values, method='lasso')
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

###########################################
## Second Plot: LARS Path (all features)
###########################################

plt.figure(figsize=(10,10))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle=None)
plt.title('LARS Path')
plt.axis('tight')
plt.legend(X_train.columns)
plt.text(fontsize=25)
plt.show()

###########################################
## Third Plot: LARS Path (main features)
###########################################

zoom_col_indices = [19, 21, 13, 15]
xxx_cols = X_train.columns[zoom_col_indices]
xxx_coefs = coefs.T[:, zoom_col_indices]

xxx = np.sum(np.abs(xxx_coefs), axis=1)
xxx /= xxx[-1]

plt.figure(figsize=(10,10))
plt.plot(xxx, xxx_coefs)
ymin, ymax = plt.ylim()
plt.vlines(xxx, ymin, ymax, linestyle=None)
plt.title('LASSO Path')
plt.axis('tight')
plt.legend(xxx_cols)
plt.show()

###########################################
## Fourth Plot: Pairplot (main features)
###########################################

plot_cols = list(xxx_cols) + ['time']

pp = sns.pairplot(balanced[plot_cols].rename(columns={
    'avg_win_time_sum': "",
    'avg_loss_time_sum': " ",
    'loss_rate_sum': "  ",
    'fastest_win_sum': "   ",
    'time': "Time"
}))
