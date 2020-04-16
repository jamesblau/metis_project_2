import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

pd.set_option('display.max_columns',100)

project_dir = "/home/james/Documents/metis/project_2/"

with open(project_dir + "second_merged.pickle", 'rb') as to_read:
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

sns.boxplot(merged['fight_count_sum'])


sns.pairplot(merged);

merged.shape

X, y = cars.drop('price',axis=1), cars['price']

# hold out 20% of the data for final testing
X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=10)
