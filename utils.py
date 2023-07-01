import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def encode_yes_no(column):
    unique_values = column.unique()
    yes_no_values = ['Yes', 'No']
    unexpected_values = set(unique_values) - set(map(str, yes_no_values))
    if unexpected_values:
        raise ValueError(
            f"Unexpected values found in column: {unexpected_values}")
    return column.map({'Yes': 1, 'No': 0})


def preprocess_dataset(dataset):
    smoke_mapping = {"Yes": 1, "No": 0, "Oc": 0.5}
    consume_mapping = {"Yes": 1, "No": 0, "Oc": 0.5, "NA": np.nan}
    headache_mapping = {
        "Occassionally": 0.5,
        "Never": 0,
        "Frequently": 1,
        np.nan: np.nan
    }

    sleep_prob_encoded = pd.get_dummies(
        dataset['Sleep_prob'], prefix='Sleep_prob')
    dataset_encoded = pd.concat([dataset, sleep_prob_encoded], axis=1)
    dataset_encoded.drop('Sleep_prob', axis=1, inplace=True)

    dataset_encoded['sleep_hrs'].fillna(
        dataset_encoded['sleep_hrs'].mean(), inplace=True)
    dataset_encoded['Headache'] = dataset_encoded['Headache'].map(
        headache_mapping)
    dataset_encoded['Smoke'] = dataset_encoded['Smoke'].map(smoke_mapping)
    dataset_encoded['Consume_TnC'] = dataset_encoded['Consume_TnC'].map(
        consume_mapping)
    dataset_encoded['Snoring'] = encode_yes_no(dataset_encoded['Snoring'])
    dataset_encoded['Participate_sport'] = encode_yes_no(
        dataset_encoded['Participate_sport'])
    dataset_encoded['Exercise'] = encode_yes_no(dataset_encoded['Exercise'])
    dataset_encoded['Headache'].fillna(
        dataset_encoded['Headache'].mean(), inplace=True)
    dataset_encoded['Consume_TnC'].fillna(
        dataset_encoded['Consume_TnC'].mean(), inplace=True)
    dataset_encoded['BMI'].fillna(dataset_encoded['BMI'].mean(), inplace=True)
    dataset_encoded['Age'].fillna(dataset_encoded['Age'].mean(), inplace=True)
    dataset_encoded['studyhrs'].fillna(
        dataset_encoded['studyhrs'].mean(), inplace=True)

    X = dataset_encoded[['BMI', 'Age', 'Meal', 'Smoke', 'Consume_TnC', 'Snoring', 'studyhrs',
                         'Exercise', 'Sleep_prob_Anxiety', 'Sleep_prob_Stress', 'Sleep_prob_Migraine']]
    y = dataset_encoded['Headache']

    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def load_model():
    model = tf.keras.models.load_model('logistic_reg_model.h5')
    return model


Smoke_mapping = {'Yes': 1, 'No': 0, 'Oc': 0.5}
Consume_TnC_mapping = {'Yes': 1, 'No': 0, 'Oc': 0.5, 'NA': np.nan}
Snoring_mapping = {'Yes': 1, 'No': 0}
Sleep_prob_mapping = {'Yes': 1, 'No': 0}


def encode_values(BMI, Age, Meal, Smoke, Consume_TnC, Snoring, studyhrs, Exercise, Sleep_prob_Anxiety,
                  Sleep_prob_Stress, Sleep_prob_Migraine):
    Smoke = Smoke_mapping.get(Smoke, 0)
    Consume_TnC = Consume_TnC_mapping.get(Consume_TnC, 0)
    Snoring = Snoring_mapping.get(Snoring, 0)
    Exercise = Smoke_mapping.get(Exercise, 0)
    Sleep_prob_Anxiety = Sleep_prob_mapping.get(Sleep_prob_Anxiety, 0)
    Sleep_prob_Stress = Sleep_prob_mapping.get(Sleep_prob_Stress, 0)
    Sleep_prob_Migraine = Sleep_prob_mapping.get(Sleep_prob_Migraine, 0)

    encoded_values = {
        'BMI': BMI, 'Age': Age, 'Meal': Meal, 'Smoke': Smoke, 'Consume_TnC': Consume_TnC, 'Snoring': Snoring,
        'studyhrs': studyhrs, 'Exercise': Exercise, 'Sleep_prob_Anxiety': Sleep_prob_Anxiety,
        'Sleep_prob_Stress': Sleep_prob_Stress, 'Sleep_prob_Migraine': Sleep_prob_Migraine
    }

    return encoded_values
