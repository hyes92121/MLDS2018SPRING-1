import numpy as np

train_x = []
train_y = []
test_x = []
test_y = []

# sample from -10 to 10
train_x = np.random.random_sample((1, 100)) * 20 - 10
train_y = np.sinc(train_x)
train = np.concatenate((train_x, train_y), axis = 0)

test_x = np.random.random_sample((1, 100)) * 20 - 10
test_y = np.sinc(test_x)
test = np.concatenate((test_x, test_y), axis = 0)

np.save("1_2_2_train.npy", train)
np.save("1_2_2_test.npy", test)
