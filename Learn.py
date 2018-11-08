import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
df = pd.read_excel('ShenyangEcxel.xlsx')

print(df.tail())

# plt(df['PM_Taiyuanjie'])
# df["PM_Taiyuanjie"] = pd.to_datetime(df["PM_Taiyuanjie"],errors='ignore', format="%m/%d/%Y %I:%M:%S %p")
plt.figure()
x=df["date"]
y1 = df["PM_Taiyuanjie"]
plt.plot(x,y1)
plt.show()

print(df.values.shape)

# x1 = df["date"]["2013-1-20 19:00:00":"2015-12-31 22:00:00"]
# y1 = df["PM_Taiyuanjie"]["2013-1-20 19:00:00":"2015-12-31 22:00:00"]
# plt.plot(x1,y1)
# plt.show()
target_pred = 'PM_Taiyuanjie'
dates = 'date'
shift_days = 7
shift_steps = shift_days * 24  # Number of hours.
# Shift data in order to make how many values want to predict
df_targets = df[target_pred].shift(-shift_steps)
# print(df_targets.tail(shift_steps + 5))


# Numpy to input data to neural network
x_data = df.values[0:-shift_steps]
print(type(x_data))
print("Shape:", x_data.shape)

y_data = df_targets.values[:-shift_steps]
print(type(y_data))
print("Shape:", y_data.shape)

num_data = len(x_data)
print (num_data)
train_split = 0.8

# num of training data
num_train = int(train_split * num_data)

# num of testing data
num_test = num_data - num_train

x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

num_x_signals = x_data.shape[1]
print(num_x_signals)

num_y_signals = y_data.shape[1]
print(num_y_signals)

print("Min:", np.min(x_train))
print("Max:", np.max(x_train))