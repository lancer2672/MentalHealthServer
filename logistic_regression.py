import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from utils import encode_yes_no, preprocess_dataset, split_dataset


dataset = pd.read_csv('Students.csv')

X, y = preprocess_dataset(dataset)

X_train, X_test, y_train, y_test = split_dataset(
    X, y, test_size=0.2, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

model_logistic = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model_logistic.compile(loss=tf.losses.binary_crossentropy,
                       optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])

history_logistic = model_logistic.fit(X_train, y_train, epochs=12)

y_pred_logistic = model_logistic.predict(X_test)
y_pred_sgd = (y_pred_logistic > 0.5).astype(int)

mae_logistic = mean_absolute_error(y_test, y_pred_sgd)

model_logistic.save('logistic_reg_model.h5')

print("Mean Absolute Error (Logistic Regression):", mae_logistic)

plt.plot(history_logistic.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.show()
