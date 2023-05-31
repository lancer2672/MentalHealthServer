from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')
app = Flask(__name__)

Smoke_mapping = {'Yes': 1, 'No': 0, 'Oc': 0.5}
Consume_TnC_mapping = {'Yes': 1, 'No': 0, 'Oc': 0.5, 'NA': np.nan}
Snoring_mapping = {'Yes': 1, 'No': 0}
sleep_hrs_mapping = {'4-6hours': 5,
                     '6-7hours': 6.5, '7-8hours': 7.5, '8-10hours': 9}
studyhrs_mapping = {1.5: 0, 4: 0.33, 6: 0.67, 8.5: 1}
Sleep_prob_mapping = {'Yes': 1, 'No': 0}


@app.route('/', methods=['GET'])
def home():
    return "HELLO WORLD"


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
    print("sleep_hrs", sleep_hrs)
    # Encode giá trị
    Smoke = Smoke_mapping.get(Smoke, 0)
    Consume_TnC = Consume_TnC_mapping.get(Consume_TnC, 0)
    Snoring = Snoring_mapping.get(Snoring, 0)
    encoded_sleep_hours = sleep_hrs_mapping.get(sleep_hrs, np.nan)
    Exercise = Smoke_mapping.get(Exercise, 0)
    Sleep_prob_Anxiety = Sleep_prob_mapping.get(Sleep_prob_Anxiety, 0)
    Sleep_prob_Stress = Sleep_prob_mapping.get(Sleep_prob_Stress, 0)
    Sleep_prob_Migraine = Sleep_prob_mapping.get(Sleep_prob_Migraine, 0)
    print("encoded_sleep_hours", encoded_sleep_hours)

    data = [
        {'BMI': BMI, 'Age': Age, 'Meal': Meal, 'Smoke': Smoke, 'Consume_TnC': Consume_TnC, 'Snoring': Snoring, 'sleep_hrs': encoded_sleep_hours, 'studyhrs': studyhrs,
         'Exercise': Exercise, 'Sleep_prob_Anxiety': Sleep_prob_Anxiety, 'Sleep_prob_Stress': Sleep_prob_Stress, 'Sleep_prob_Migraine': Sleep_prob_Migraine},
    ]
    input_data1 = pd.DataFrame(data, columns=['BMI', 'Age', 'Meal', 'Smoke', 'Consume_TnC', 'Snoring', 'sleep_hrs', 'studyhrs',
                                              'Exercise', 'Sleep_prob_Anxiety', 'Sleep_prob_Stress',
                                              'Sleep_prob_Migraine'])
    # Thực hiện dự đoán cho dữ liệu params
    y_pred = y_pred = model.predict(input_data1)
    print("y_pred", y_pred)
    return jsonify({'prediction': y_pred.tolist()})


def create_app():
    return app
# if __name__ == '__main__':
#     from waitress import serve
#     serve(app, host="0.0.0.0", port=8080)
