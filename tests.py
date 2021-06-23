import numpy as np
import pandas as pd
from optimizers import SGD
import tensorflow as tf
from sklearn.model_selection import train_test_split
from algorithms import Algorithms

# ------------------ Random devices ---------------------
sgd = SGD()
'''random_device_list = sgd.random_devices()
print('Random device list:')
print(random_device_list)'''

# 940, 6D0, 160, 1C0, 220
# 160, 940, 6D0
# ------------------ TESTS: OK --------------------------

# ------------------ Random user data -------------------
random_user_data_list, random_user_output_list = sgd.random_user_data()
random_user_data = random_user_data_list[0] 
random_user_output = random_user_output_list[0]

print('Random user data: ')
print(type(random_user_data))
print(random_user_data)
print('Random user output: ')
random_user_output = np.asarray(random_user_output)
print(type(random_user_output))
print(random_user_output) # first element doesn't count

# 910, 130, 160
# ------------------ TESTS: OK --------------------------

# ------------------ Loss function with Keras----------------------
df = pd.read_csv('data.csv')
x = df.loc[:, df.columns != 'Output'].to_numpy()
y = df.loc[:, df.columns == 'Output'].to_numpy()
df['Input'] = df[df.columns[0:-1]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1) # converting Input_0 etc. into Input
df['Input'] = pd.to_numeric(df['Input'])

X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.20,random_state=42)
X_Train = X_Train.astype('int64')
X_Test = X_Test.astype('int64')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Dense(3))
model.compile(optimizer='sgd', loss='mse')
model.fit(X_Train, Y_Train, batch_size=10, epochs=100)
y_pred = model.predict(X_Train)

print('Numpy')
se = (Y_Train - y_pred)**2
mse = np.mean(se)
print(mse)
print('SGD MSE')
print(sgd.loss(Y_Train, y_pred))
# ------------------ TESTS: OK --------------------------

# --------------- Gradient Descent with own NN --------------
nn = Algorithms.NeuralNetwork()
W1, b1, W2, b2 = nn.gradient_descent(random_user_data, random_user_output, 0.1, 500)
# ------------------ TESTS: Errors ----------------------
