from keras.models import load_model

import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

import numpy as np
import pandas as pd

from ttictoc import tic,toc

loaded_model = load_model('LSTM.h5')
xtest = pd.read_csv('xtest.csv',  header=None)

times_=100




tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=1, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed = toc()
print("nothing",elapsed)
tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=1, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed = toc()
print("nothing",elapsed/times_)

tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=1, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed1 = toc()
print("Timing for one sample",elapsed1/times_)





################################### one sample

tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=5, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)
elapsed2 = toc()
print("Timing for 5 sample",elapsed2/times_)

#################################### Blocks


tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=10, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed3 = toc()
print("Timing for block of 10 sample", elapsed3/times_)


tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=15, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed4 = toc()
print("Timing for block of 15 sample", elapsed4/times_)


tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=20, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed5 = toc()
print("Timing for block of 20 sample", elapsed5/times_)

tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=25, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed6 = toc()
print("Timing for block of 25 sample", elapsed6/times_)

tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=30, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed7 = toc()
print("Timing for block of 30 sample", elapsed7/times_)


tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=35, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed8 = toc()
print("Timing for block of 35 sample", elapsed8/times_)

tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=40, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed9 = toc()
print("Timing for block of 40 sample", elapsed9/times_)

tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=45, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed10 = toc()
print("Timing for block of 45 sample", elapsed10/times_)

tic()
for i in range(times_):
    xtest_one = np.array(xtest.sample(n=50, replace=True))
    xtest_one = xtest_one.reshape(xtest_one.shape[0], 1, xtest_one.shape[1])
    prediction = loaded_model.predict(xtest_one)

elapsed11 = toc()
print("Timing for block of 50 sample", elapsed11/times_)



# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

