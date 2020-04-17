import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

with open("pickles/second_merged.pickle", 'rb') as to_read:
    merged = pickle.load(to_read)

merged['male'] = pd.get_dummies(merged['male'], drop_first=True)

# Let's not bet on tomato cans
merged = merged[merged['fight_count_sum'] < 100]

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

# balanced.shape

# sns.boxplot(balanced['fight_count_sum'])

# balanced[cols1]

sns.pairplot(balanced[cols1]);

# balanced.shape

#####################

sns.set()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score

cols1 = ['fight_count', 'height', 'class', 'male', 'fight_count_2', 'time']
cols2 = ['height_2', 'fight_count_dif', 'fight_count_sum', 'height_dif', 'height_sum', 'time']
cols3 = ['win_rate_dif', 'win_rate_sum', 'loss_rate_dif', 'loss_rate_sum', 'fastest_win_dif', 'time']
cols4 = ['fastest_win_sum', 'fastest_loss_dif', 'fastest_loss_sum', 'avg_win_time_dif', 'time']
cols5 = ['avg_win_time_sum', 'avg_loss_time_dif', 'avg_loss_time_sum', 'fights_quantile', 'time']
sumcols = [col for col in balanced.columns if 'sum' in col]
difcols = [col for col in balanced.columns if 'dif' in col]

X = balanced.drop(['time'], 1)
Xsum = balanced.drop(difcols + ['time'], 1)
Xdif = balanced.drop(sumcols + ['time'], 1)
y = balanced['time']

# cols = ['fight_count', 'height', 'class', 'male', 'fight_count_2', 'time']
cols = ['height_2', 'fight_count_dif', 'fight_count_sum', 'height_dif', 'height_sum', 'time']
# cols = ['win_rate_dif', 'win_rate_sum', 'loss_rate_dif', 'loss_rate_sum', 'fastest_win_dif', 'time']
# cols = ['fastest_win_sum', 'fastest_loss_dif', 'fastest_loss_sum', 'avg_win_time_dif', 'time']
# cols = ['avg_win_time_sum', 'avg_loss_time_dif', 'avg_loss_time_sum', 'fights_quantile', 'time']
sns.pairplot(balanced[cols])

#####################

X = Xdif

X_train_val, X_test, y_train_val, y_test = \
        train_test_split(X, y, test_size=0.2,random_state=42)
X_train, X_val, y_train, y_val = \
        train_test_split(X_train_val, y_train_val, test_size=.25, random_state=43)

## Scale the data as before
std = StandardScaler()
std.fit(X_train.values)

## Scale the Predictors on both the train and test set
X_tr = std.transform(X_train.values)
X_te = std.transform(X_test.values)

# Run the cross validation, find the best alpha, refit the model on all the data with that alpha

alphavec = 10**np.linspace(-2,2,200)

lasso_model = LassoCV(alphas = alphavec, cv=5)
lasso_model.fit(X_tr, y_train)

# This is the best alpha value it found - not far from the value
# selected using simple validation
lasso_model.alpha_

# These are the (standardized) coefficients found
# when it refit using that best alpha
list(zip(X_train.columns, lasso_model.coef_))

# Make predictions on the test set using the new model
test_set_pred = lasso_model.predict(X_te)

# Mean Absolute Error (MAE)
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

# Find the MAE and R^2 on the test set using this model
mae(y_test, test_set_pred) # 4 min 26 seconds
r2_score(y_test, test_set_pred)

#####################

help(df\
    .groupby(['f1'])\
    .agg({
        'fight_count': 'first',
    }).hist)

# PLOT
fight_counts_agg = df\
    .groupby(['f1'])\
    .agg({
        'fight_count': 'first',
    })

fight_counts_agg.hist()

# Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# sns.boxplot(fight_counts_agg)



#####################

from sklearn.linear_model import lars_path

## Scale the variables
std = StandardScaler()
std.fit(X_train.values)

X_tr = std.transform(X_train.values)

## Note: lars_path takes numpy matrices, not pandas dataframes

print("Computing regularization path using the LARS ...")
alphas, _, coefs = lars_path(X_tr, y_train.values, method='lasso')

# plotting the LARS path

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

len(xx)
len(X_train.columns)
coefs.T.shape

# PLOT
plt.figure(figsize=(10,10))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle=None)
# plt.xlabel('|coef| / max|coef|')
# plt.ylabel('Coefficients')
plt.title('LARS Path')
plt.axis('tight')
plt.legend(X_train.columns)
plt.text(fontsize=25)
plt.show()

##

zoom_col_indices = [9, 11, 13, 15]
zoom_col_indices = [19, 21, 13, 15]
xxx_cols = X_train.columns[zoom_col_indices]
xxx_coefs = coefs.T[:, zoom_col_indices]

xxx = np.sum(np.abs(xxx_coefs), axis=1)
xxx /= xxx[-1]

len(xxx)
len(xxx_cols)
xxx_coefs.shape

# PLOT

plt.figure(figsize=(10,10))
plt.plot(xxx, xxx_coefs)
ymin, ymax = plt.ylim()
plt.vlines(xxx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.legend(xxx_cols)
plt.show()

coefs.T[-1]

##

plot_cols = list(xxx_cols) + ['time']
plot_cols

# PLOT
pp = sns.pairplot(balanced[plot_cols].rename(columns={
        'avg_win_time_sum': "",
        'avg_loss_time_sum': " ",
        'loss_rate_sum': "  ",
        'fastest_win_sum': "   ",
        'time': "Time"
    }))

     x_vars=["sepal_width", "sepal_length"],
     y_vars=["petal_width", "petal_length"])

help(sns.pairplot)

#####################

# X, y = balanced[['fight_count_dif', 'fastest_loss_sum', 'class']], balanced['time']

# # hold out 20% of the data for final testing
# X_train, X_val, y_train, y_val = \
        # train_test_split(X, y, test_size=.25, random_state=3)

# #set up the 3 models we're choosing from:

# lm = LinearRegression()

# #Feature scaling for train, val, and test so that we can run our ridge model on each
# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train.values)
# X_val_scaled = scaler.transform(X_val.values)
# X_test_scaled = scaler.transform(X_test.values)

# lm_reg = Ridge(alpha=1)

# #Feature transforms for train, val, and test so that we can run our poly model on each
# poly = PolynomialFeatures(degree=2)

# X_train_poly = poly.fit_transform(X_train.values)
# X_val_poly = poly.transform(X_val.values)
# X_test_poly = poly.transform(X_test.values)

# lm_poly = LinearRegression()

# #validate

# lm.fit(X_train, y_train)
# print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')

# lm_reg.fit(X_train_scaled, y_train)
# print(f'Ridge Regression val R^2: {lm_reg.score(X_val_scaled, y_val):.3f}')

# lm_poly.fit(X_train_poly, y_train)
# print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')

# #####################

# classes = merged['class'].unique()

# [merged[merged['class'] == clss]['time'].mean() for clss in sorted(classes)]

# #####################

# from sklearn.model_selection import cross_val_score
# lm = LinearRegression()

# cross_val_score(lm, X, y, # estimator, features, target
                # cv=5, # number of folds
                # scoring='r2') # scoring metric

# kf = KFold(n_splits=5, shuffle=True, random_state = 71)
# cross_val_score(lm, X, y, cv=kf, scoring='r2')

# kf = KFold(n_splits=5, shuffle=True, random_state = 1000)

# print(np.mean(cross_val_score(lm, X, y, cv=kf, scoring='r2')))
# print(np.mean(cross_val_score(lm_reg, X, y, cv=kf, scoring='r2')))

# #####################

# sns.set()

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
# from sklearn.metrics import r2_score
