from flask import Flask, request, jsonify
import numpy as np
from linear_model import regression, mean_squared_error, r2_score
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "HELLO WORLD"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json


Meal_mapping = {1: 0, 2: 0.33, 3: 0.67, 4: 1}
Smoke_mapping = {'Yes': 1, 'No': 0, 'Oc': 0.5}
Consume_TnC_mapping = {'Yes': 1, 'No': 0, 'Oc': 0.5, 'NA': np.nan}
Snoring_mapping = {'Yes': 1, 'No': 0}
sleep_hrs_mapping = {'4-6hours': 1,
                     '6-7hours': 0, '7-8hours': 0, '8-10hours': 1}
studyhrs_mapping = {1.5: 0, 4: 0.33, 6: 0.67, 8.5: 1}
Sleep_prob_mapping = {'Yes': 1, 'No': 0}


@app.route('/get-predicted-values', methods=['GET'])
def get_predicted_values():
    params = request.args
    print(params)

    # Trích xuất các giá trị params
    BMI = float(params.get('BMI', 0))
    Age = int(params.get('Age', 0))
    Meal = int(params.get('Meal', 0))
    Smoke = (params.get('Smoke', "No"))
    Consume_TnC = (params.get('Consume_TnC', "No"))
    Snoring = (params.get('Snoring', "No"))
    sleep_hrs = (params.get('sleep_hrs', "NA"))
    studyhrs = float(params.get('studyhrs', "NA"))
    Exercise = (params.get('Exercise', "No"))
    Sleep_prob_Anxiety = (params.get('Sleep_prob_Anxiety', "No"))
    Sleep_prob_Stress = (params.get('Sleep_prob_Stress', "No"))
    Sleep_prob_Migraine = (params.get('Sleep_prob_Migraine', "No"))
    print("SLEEP HOURS", sleep_hrs)
    # Encode giá trị
    Meal = Meal_mapping.get(Meal, 0)
    Smoke = Smoke_mapping.get(Smoke, 0)
    Consume_TnC = Consume_TnC_mapping.get(Consume_TnC, 0)
    Snoring = Snoring_mapping.get(Snoring, 0)
    encoded_sleep_hours = sleep_hrs_mapping.get(sleep_hrs, np.nan)
    studyhrs = studyhrs_mapping.get(studyhrs, np.nan)
    print("SLEEP HOURS encoded_sleep_hours", encoded_sleep_hours)
    Exercise = Smoke_mapping.get(Exercise, 0)
    Sleep_prob_Anxiety = Sleep_prob_mapping.get(Sleep_prob_Anxiety, 0)
    Sleep_prob_Stress = Sleep_prob_mapping.get(Sleep_prob_Stress, 0)
    Sleep_prob_Migraine = Sleep_prob_mapping.get(Sleep_prob_Migraine, 0)

    data = [
        {'BMI': BMI, 'Age': Age, 'Meal': Meal, 'Smoke': Smoke, 'Consume_TnC': Consume_TnC, 'Snoring': Snoring, 'sleep_hrs': encoded_sleep_hours, 'studyhrs': studyhrs,
         'Exercise': Exercise, 'Sleep_prob_Anxiety': Sleep_prob_Anxiety, 'Sleep_prob_Stress': Sleep_prob_Stress, 'Sleep_prob_Migraine': Sleep_prob_Migraine},
    ]
    input_data1 = pd.DataFrame(data, columns=['BMI', 'Age', 'Meal', 'Smoke', 'Consume_TnC', 'Snoring', 'sleep_hrs', 'studyhrs',
                                              'Exercise', 'Sleep_prob_Anxiety', 'Sleep_prob_Stress',
                                              'Sleep_prob_Migraine'])
    # input_data = [{BMI, Age, Meal, Smoke, Consume_TnC, Snoring, sleep_hrs, studyhrs,
    #                Exercise, Sleep_prob_Anxiety, Sleep_prob_Stress, Sleep_prob_Migraine}]
    # input_data1 = pd.DataFrame(input_data, columns=['BMI', 'Age', 'Meal', 'Smoke', 'Consume_TnC', 'Snoring', 'sleep_hrs', 'studyhrs', 'Exercise', 'Sleep_prob_Anxiety', 'Sleep_prob_Stress', 'Sleep_prob_Migraine'])

    print("INPUT DATA", input_data1)
    print("INPUT DATA", input_data1.columns)
    print("INPUT DATA", input_data1.values)
    # Thực hiện dự đoán cho dữ liệu params
    y_pred = regression.predict(input_data1)
    print("y_pred", y_pred)

    return jsonify({'prediction': y_pred.tolist()})


if __name__ == '__main__':
    app.run()
