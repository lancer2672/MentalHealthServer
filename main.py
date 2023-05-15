from fastapi import FastAPI
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
data = pd.read_csv("csv.csv")

data.shape
data.plot(kind='scatter', x="SleepValue", y="MentalValue")
sleepValue = pd.DataFrame(data['SleepValue'])
mentalValue = pd.DataFrame(data['MentalValue'])
lm = linear_model.LinearRegression()
model = lm.fit(mentalValue, sleepValue)
newSleepValue = 8
newMentalValue = model.predict(np.array([[newSleepValue]]))
# print("model.coef", model.coef_)
# print("model.intercept", model.intercept_)
# print("modelscore", model.score(mentalValue, sleepValue))
# print("newMentalValue", newMentalValue)

# plt.plot(mentalValue, model.predict(mentalValue), color='red')
# plt.show()


app = FastAPI()

print(data)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items")
async def predictMentalValue():
    return {"message": float(newMentalValue)}
