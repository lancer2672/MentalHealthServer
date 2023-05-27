import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

dataset = pd.read_csv('Students.csv')
# Convert 'Yes' and 'No' values to 1 and 0 in the 'Smoke', 'Consume_T&C', 'Snoring', 'Participate_sport', and 'Exercise' columns
yes_no_values = ['Yes', 'No']
sleep_hours_mapping = {
    '4-6hours': 1,
    '6-7hours': 0,
    '7-8hours': 0,
    '8-10hours': 1,
}
study_hours_mapping = {
    1.5: 0,
    4: 0.33,
    6: 0.67,
    8.5: 1
}
smoke_mapping = {"Yes": 1, "No": 0, "Oc": 0.5}
consume_mapping = {"Yes": 1, "No": 0, "Oc": 0.5, "NA": np.nan}
headache_mapping = {"Occassionally": 0.5,

                    "Never": 0,
                    "Frequently": 1,
                    np.nan: np.nan}


def encode_yes_no(column):
    unique_values = column.unique()
    unexpected_values = set(unique_values) - set(yes_no_values)
    if unexpected_values:
        raise ValueError(
            f"Unexpected values found in column: {unexpected_values}")
    return column.map({'Yes': 1, 'No': 0})


def convert_height(value):
    if isinstance(value, str):
        if value == "NA":
            return np.nan
        else:
            low, high = map(int, value.split("-"))
            return (low + high) / 2
    else:
        return value


dataset['sleep_hrs'] = dataset['sleep_hrs'].map(sleep_hours_mapping)
# Perform one-hot encoding on 'Sleep_prob'
sleep_prob_encoded = pd.get_dummies(dataset['Sleep_prob'], prefix='Sleep_prob')

# Concatenate the encoded columns with the original dataset
dataset_encoded = pd.concat([dataset, sleep_prob_encoded], axis=1)

# parsev values
dataset_encoded.drop('Sleep_prob', axis=1, inplace=True)
dataset_encoded['Height'] = dataset_encoded['Height'].apply(convert_height)


# Fill in empty spaces with mean value of Height and Weight

dataset_encoded['sleep_hrs'].fillna(
    dataset_encoded['sleep_hrs'].mean(), inplace=True)
dataset_encoded['Headache'] = dataset_encoded['Headache'].map(headache_mapping)
dataset_encoded['Smoke'] = dataset_encoded['Smoke'].map(smoke_mapping)
dataset_encoded['studyhrs'] = dataset_encoded['studyhrs'].map(
    study_hours_mapping)
dataset_encoded['Consume_T&C'] = dataset_encoded['Consume_T&C'].map(
    consume_mapping)

dataset_encoded['Snoring'] = encode_yes_no(dataset_encoded['Snoring'])
dataset_encoded['Participate_sport'] = encode_yes_no(
    dataset_encoded['Participate_sport'])
dataset_encoded['Exercise'] = encode_yes_no(dataset_encoded['Exercise'])

# Convert 'Yes' and 'No' values to 1 and 0 in the 'Smoke', 'Consume_T&C', 'Snoring', 'Participate_sport', and 'Exercise' columns

dataset_encoded['Headache'].fillna(
    dataset_encoded['Headache'].mean(), inplace=True)
dataset_encoded['Consume_T&C'].fillna(
    dataset_encoded['Consume_T&C'].mean(), inplace=True)
dataset_encoded['Height'].fillna(
    dataset_encoded['Height'].mean(), inplace=True)
dataset_encoded['Weight'].fillna(
    dataset_encoded['Weight'].mean(), inplace=True)
dataset_encoded['BMI'].fillna(
    dataset_encoded['BMI'].mean(), inplace=True)
dataset_encoded['Age'].fillna(
    dataset_encoded['Age'].mean(), inplace=True)
dataset_encoded['studyhrs'].fillna(
    dataset_encoded['studyhrs'].mean(), inplace=True)
# X = dataset_encoded[['BMI', 'Age', 'Meal', 'Smoke', 'Height', 'Weight', 'sleep_hrs', 'Consume_T&C', 'Snoring', 'studyhrs',
#                      'Participate_sport', 'Exercise', 'Sleep_prob_None', 'Sleep_prob_Depression/Anxiety', 'Sleep_prob_Stress', 'Sleep_prob_Migraine']]

X = dataset_encoded[['BMI', 'Age', 'Meal', 'Smoke',  'Consume_T&C', 'Snoring', 'sleep_hrs', 'studyhrs',
                     'Exercise',  'Sleep_prob_Depression/Anxiety', 'Sleep_prob_Stress', 'Sleep_prob_Migraine']]

y = dataset_encoded['Headache']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a linear regression object
regression = LinearRegression()

# Train the model using the training sets
regression.fit(X_train, y_train)

# Predict on the test data
y_pred = regression.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate coefficient of determination (R-squared)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("Coefficient of determination (R-squared): ", r2)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r')
plt.show()
