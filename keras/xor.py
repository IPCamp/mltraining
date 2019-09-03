from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.initializers import RandomUniform

rndU = RandomUniform(minval=-1, maxval=1, seed=None)

model = Sequential()
model.add(Dense(9, activation='sigmoid', input_dim=2, use_bias = True, kernel_initializer=rndU, bias_initializer=rndU))
model.add(Dense(1, activation='sigmoid', use_bias = True, kernel_initializer=rndU, bias_initializer=rndU))

sgd = SGD(lr=1)

model.compile(optimizer=sgd,
              loss='mse',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 1, 1, 0])


# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(inputs, outputs, epochs=1000, batch_size=1)