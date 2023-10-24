import pandas as pd

schedule = pd.read_csv(r"C:/Users/tyleg/Desktop/miniprojects/wnba/reg_season.csv")
schedule = schedule.iloc[:, 1:-2]
#print(schedule.head())

advanced_stats = pd.read_csv(r"C:/Users/tyleg/Desktop/miniprojects/wnba/advanced_stats.csv")
advanced_stats = advanced_stats.dropna(axis=1, how='all')
advanced_stats = advanced_stats.iloc[:, 1:-1]
#print(advanced_stats.head())

df = pd.merge(schedule, advanced_stats, left_on="Visitor/Neutral", right_on="Team")
df = pd.merge(df, advanced_stats, left_on="Home/Neutral", right_on="Team")
df = df.drop(['Team_x', 'Team_y'], axis=1)
#print(df.head())

for index, row in df.iterrows():
    if df.loc[index, 'PTS'] > df.loc[index, 'PTS.1']:
        df.loc[index, 'Home_Winner'] = 0
    else:
        df.loc[index, 'Home_Winner'] = 1
#print(df.head())

# Determine which columns we don't want to use for our prediction model

# Columns we want to remove
remove_cols = ["Home/Neutral", "Visitor/Neutral", "Home_Winner", "PTS", "PTS.1"]

# Columns we want to keep
selected_cols = [x for x in df.columns if x not in remove_cols]

# Display columns we want to keep
#print(df[selected_cols].head())


# Scale our data
from sklearn.preprocessing import MinMaxScaler

# Initialize scaler
scalar = MinMaxScaler()
# Scale our data
df[selected_cols] = scalar.fit_transform(df[selected_cols])
# Display dataframe
#print(df.head())

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier

# Initialize our ridge regression classification
rr = RidgeClassifier(alpha=1.0)

# Initialize our feature selector which picks the best 10 features backward
sfs = SequentialFeatureSelector(rr, n_features_to_select=10, direction='backward')

# Determine which columns are the most impactful when predicting the winner
sfs.fit(df[selected_cols], df['Home_Winner'])

# Create a list of the most impactful columns
predictors = list(df[selected_cols].columns[sfs.get_support()])
# Display the most impactful columns
#print(df[predictors].head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def monte_carlo(n):
    accuracy = []
    for i in range(n):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(df[predictors], df['Home_Winner'], test_size=0.2)

        # Train a logistic regression model on the training data
        model = LogisticRegression()
        # Fit the model to our training data
        model.fit(X_train, y_train)

        # Predict the winners for the test data
        y_pred = model.predict(X_test)

        # Evaluate the accuracy of the model on the test data
        accuracy.append(accuracy_score(y_test, y_pred))

    # Get the average accuracy
    score = sum(accuracy) / len(accuracy)
    return score

score = monte_carlo(1000)
print(f"Accuracy: {score}")

# Remove the rows where the Aces and Liberty play against each other
non_aces_liberty_game = df
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')].index)
non_aces_liberty_game = non_aces_liberty_game.drop(non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'New York Liberty') & (non_aces_liberty_game['Visitor/Neutral'] == 'Las Vegas Aces')].index)
non_aces_liberty_game[(non_aces_liberty_game['Home/Neutral'] == 'Las Vegas Aces') & (non_aces_liberty_game['Visitor/Neutral'] == 'New York Liberty')]

# Get a game where the Aces are home and Liberty is away
final_matchup = df[(df['Home/Neutral'] == 'Las Vegas Aces') & (df['Visitor/Neutral'] == 'New York Liberty')][:1]
# Show the predictors we will use
final_matchup[predictors]

# Predict the winner of the final matchup
model = LogisticRegression()
model.fit(non_aces_liberty_game[predictors], non_aces_liberty_game['Home_Winner'])

# Predict the outcome of the final_matchup
y_pred = model.predict(final_matchup[predictors])
print(f"Prediction: {y_pred[0]}")