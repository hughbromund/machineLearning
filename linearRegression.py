import numpy
import keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Downloads all of the boston_housing data

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

def baseline_model():
    model = Sequential()
    model.add(Dense(50, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    # model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = baseline_model()

model.fit(x_train, y_train, epochs=1500, batch_size=15, validation_data=(x_test, y_test), verbose=2, shuffle=True)

