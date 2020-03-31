import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('./tennis_stats.csv')
# print(df.head())
columns = df.columns
# print(df.describe())
# print(columns)


# # perform exploratory analysis here:
# plt.scatter(df['Losses'], df['Winnings'])
# plt.show()



# perform single feature linear regressions here:
# features = df[['Wins']]
# outcome = df[['Winnings']]

# #Using train_test_split to split data into training and test sets
# features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8)

# #Creating linear regression object
# mlr = LinearRegression()
# mlr.fit(features_train, outcome_train)

# #Scoring and predicting outcomes
# print('Predicted Winnings with Wins test score: ', mlr.score(features_test, outcome_test))
# outcome_predict = mlr.predict(features_test)
# plt.scatter(outcome_test, outcome_predict, alpha=0.4)
# plt.show()
# plt.clf()



## Creating a function to replicate the functionality above
def single_feature_regression(feature, result):
    category = df[[feature]]
    results = df[[result]]

    category_train, category_test, results_train, results_test = train_test_split(category, results, train_size=0.8)

    model = LinearRegression()
    model.fit(category_train, results_train)

    print('Predicted ' + result + ' with ' + feature + ' test score: ', model.score(category_test, results_test))

    predict_results = model.predict(category_test)

    plt.scatter(results_test, predict_results, alpha = 0.4)
    plt.title('Predicted Results vs. Actual Results')
    plt.xlabel('Actual ' + result)
    plt.ylabel('Predicted ' + result)
    plt.show()
    plt.clf()

#Calling function with string of data column
# single_feature_regression('BreakPointsOpportunities', 'Winnings')


## perform two feature linear regressions here:
# Creating a function to handle two feature lin regression
def two_feature_regression(feature1, feature2, result):
    categories = df[[feature1, feature2]]
    results = df[[result]]

    category_train, category_test, results_train, results_test = train_test_split(categories, results, train_size=0.8)

    model = LinearRegression()
    model.fit(category_train, results_train)

    print('Predicted ' + result + ' with two features test score: ', model.score(category_test, results_test))

    predict_results = model.predict(category_test)

    plt.scatter(results_test, predict_results, alpha = 0.4)
    plt.title('Predicted Results vs. Actual Results')
    plt.xlabel('Actual ' + result)
    plt.ylabel('Predicted ' + result)
    plt.show()
    plt.clf()

# two_feature_regression('BreakPointsOpportunities', 'FirstServeReturnPointsWon', 'Winnings')



## Performing multiple feature linear regressions to predict winnings
categories = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon']]
results = df[['Winnings']]
category_train, category_test, results_train, results_test = train_test_split(categories, results, train_size=0.8)
model = LinearRegression()
model.fit(category_train, results_train)
print('Predicted Winnings with two features test score: ', model.score(category_test, results_test))
predict_results = model.predict(category_test)
plt.scatter(results_test, predict_results, alpha = 0.4)
plt.title('Predicted Results vs. Actual Results')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()

# I attempted to create a function for this as well but ran into a string conversion to float issue when inputting a list of all features in


