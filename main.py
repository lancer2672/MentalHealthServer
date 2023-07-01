from flask import Flask, request, jsonify
from utils import load_model, encode_values
import pandas as pd

app = Flask(__name__)
model = load_model()


@app.route('/', methods=['GET'])
def home():
    return "HELLO WORLD"


@app.route('/get-predicted-values', methods=['GET'])
def get_predicted_values():
    params = request.args

    # Trích xuất các giá trị params
    BMI = float(params.get('BMI', 0))
    Age = int(params.get('Age', 0))
    Meal = int(params.get('Meal', 0))
    Smoke = params.get('Smoke', "No")
    Consume_TnC = params.get('Consume_TnC', "No")
    Snoring = params.get('Snoring', "No")

    studyhrs = float(params.get('studyhrs', "NA"))
    Exercise = params.get('Exercise', "No")
    Sleep_prob_Anxiety = params.get('Sleep_prob_Anxiety', "No")
    Sleep_prob_Stress = params.get('Sleep_prob_Stress', "No")
    Sleep_prob_Migraine = params.get('Sleep_prob_Migraine', "No")

    # Encode giá trị
    encoded_values = encode_values(BMI, Age, Meal, Smoke, Consume_TnC, Snoring, studyhrs, Exercise,
                                   Sleep_prob_Anxiety, Sleep_prob_Stress, Sleep_prob_Migraine)

    data = [encoded_values]
    input_data = pd.DataFrame(data, columns=['BMI', 'Age', 'Meal', 'Smoke', 'Consume_TnC', 'Snoring',
                                             'studyhrs', 'Exercise', 'Sleep_prob_Anxiety', 'Sleep_prob_Stress',
                                             'Sleep_prob_Migraine'])

    # Thực hiện dự đoán cho dữ liệu params
    y_pred = model.predict(input_data)

    return jsonify({'prediction': y_pred.tolist()})


def create_app():
    return app


if __name__ == '__main__':
    app.run(host="0.0.0.0")
