import numpy as np
import pandas as pd
import math, time
import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt2
from os import makedirs
from os.path import exists, join

class file_logger():
    def __init__(self, file_path, append=True):
        self.file_path = file_path

        if not append:
            open(self.file_path, 'w').close()

    def __call__(self, string):
        str_aux = string + '\n'

        f = open(self.file_path, 'a')
        f.write(str_aux)
        f.close()

def get_stock_data(stock_name, normalized=0):

    col_names = ['', 'v', 'vw', 'o', 'c', 'h', 'I', 't',
                 'n']
    stocks = pd.read_csv(stock_name, header = 0, names=col_names)
    df = pd.DataFrame(stocks)
    return df

stock_name = '/home/xijun.wang/CS496_DL_Seminar/data/historical_price/AAL_2015-12-30_2021-02-21_minute.csv'
comany_name = 'AAL'
df = get_stock_data(stock_name, 0)
df.head()
print(df.head())

df.drop(df.columns[[0, 1, 2, 3, 5, 6, 7, 8]], axis=1, inplace=True)
df.head()
print(df.head())

path_save = '/home/xijun.wang/CS496_DL_Seminar/data/pure_close_price/'
today = datetime.date.today()
filename = path_save+'stock_%s.csv' % today
df.to_csv(filename)

df.head()
print(df.head())

#Load the data
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.8 * result.shape[0])
    row_1 = round(3/4 * row) # 60% train, 20% val
    train = result[:int(row_1), :]
    val = result[int(row_1):int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_val = val[:, :-1]
    y_val = val[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_val, y_val, x_test, y_test]

# Build the model
def build_model(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

# Input: actual close price sequence (N, ), predicted close price sequence (N, 1), epsilon_1 (scalar), epsilon_2 (scalar)
# Output: Total gain and gain percent, buy_time, buy_price, sell_time, sell_price
# Purpose: Compute the gain (percent) during trading
# even though the function is named hill_climbing, but it is actually the trading strategy process
def hill_climbing(C_actual, P_predict, epsilon_1, epsilon_2):
    # record the buy and sell price and time
    buy_time = []
    buy_price = []
    sell_time = []
    sell_price = []
    flag = 0 # control buy-sell operations, 0 stands for "at buy state", 1 for "at sell state"
    s = 0
    s -= C_actual[0]
    buy_time.append(0)
    buy_price.append(C_actual[0])
    for index in range(len(C_actual) - 1):
        # sell
        if flag == 0 and C_actual[index] - P_predict[index+1][0] > epsilon_2:
            flag = 1
            s += C_actual[index]
            sell_time.append(index)
            sell_price.append(C_actual[index])
        # buy
        if flag == 1 and P_predict[index+1][0] - C_actual[index] > epsilon_1:
            flag = 0
            s -= C_actual[index]
            buy_time.append(index)
            buy_price.append(C_actual[index])
    if flag == 1:
        g = s
    else:
        g = s + C_actual[-1]
        sell_time.append(len(C_actual) - 1)
        sell_price.append(C_actual[-1])
    g = round(float(g), 8)
    pg = g / C_actual[0] * 100
    return g, pg, buy_time, buy_price, sell_time, sell_price



window = 30
X_train, y_train, X_val, y_val, X_test, y_test = load_data(df[::-1], window)

print("X_train", X_train.shape) # (2782, 30, 1)
print("y_train", y_train.shape) # (2782,)
print("X_val", X_val.shape) # (2782, 30, 1)
print("y_val", y_val.shape) # (2782,)
print("X_test", X_test.shape) # (695, 30, 1)
print("y_test", y_test.shape) # (695,)

model = build_model([1, window, 1])
model.summary()

model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=50,
    verbose=1)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

# predicted validation and test
p_val = model.predict(X_val)
p_test = model.predict(X_test)


max = 1 / 4 * (np.amax(y_val) - np.amin(y_val))
min = 0
mu, sigma = 1 / 2 * (max - min), 1 / 4 * (max - min)  # mean and standard deviation

pg_test_list = []
epsilon_1_test_list = []
epsilon_2_test_list = []

# path_save_result_images = '/Users/xi/Downloads/{}/'.format(comany_name)
path_save_result_images = '/home/xijun.wang/CS496_DL_Seminar/image_results/{}/'.format(comany_name)
if not exists(path_save_result_images):
    makedirs(path_save_result_images)

loggers = file_logger(path_save_result_images+'log.txt', False)

loggers('Test Score: {} MSE ({} RMSE) \n'.format(testScore[0], math.sqrt(testScore[0])))

# run for 10 times due to the statistic nature of HC
for i in range(10):
    epsilon_1_array = np.random.normal(mu, sigma, 60) # (60, )
    epsilon_2_array = np.random.normal(mu, sigma, 60)
    # Using the HC algorithm in charge of setting the optimal thresholds for buying and selling runs for 60 iterations
    for index in range(len(epsilon_1_array)):
        if epsilon_1_array[index] < min:
            epsilon_1_array[index] = min
        if epsilon_1_array[index] > max:
            epsilon_1_array[index] = max
        if epsilon_2_array[index] < min:
            epsilon_2_array[index] = min
        if epsilon_2_array[index] > max:
            epsilon_2_array[index] = max

    g_val_list = []
    pg_val_list = []

    for index in range(len(epsilon_1_array)):
        g, pg, _, _, _, _ = hill_climbing(y_val, p_val, epsilon_1_array[index], epsilon_2_array[index])
        g_val_list.append(g)
        pg_val_list.append(pg)

    g_val_array = np.array(g_val_list)
    max_gain_index = g_val_array.argmax()

    # get the optimal epsilon_1 and epsilon_2
    epsilon_1 = epsilon_1_array[max_gain_index]
    epsilon_2 = epsilon_2_array[max_gain_index]

    # print("p shape", p.shape) # (695, 1)
    g_test, pg_test, buy_time, buy_price, sell_time, sell_price = hill_climbing(y_test, p_test, epsilon_1, epsilon_2)
    # print("gain for testing:", g_test)
    # print("gain percent for testing:", pg_test)
    pg_test_list.append(pg_test)
    epsilon_1_test_list.append(epsilon_1)
    epsilon_2_test_list.append(epsilon_2)

    if i == 0:
        pg_test_max = pg_test
        max_test_gain_percent_index = 0
        buy_time_max = buy_time
        buy_price_max = buy_price
        sell_time_max = sell_time
        sell_price_max = sell_price
    else:
        if pg_test > pg_test_max:
            pg_test_max = pg_test
            max_test_gain_percent_index = i
            buy_time_max = buy_time
            buy_price_max = buy_price
            sell_time_max = sell_time
            sell_price_max = sell_price

pg_test_array = np.array(pg_test_list)
max_test_gain_percent_index = pg_test_array.argmax()
min_test_gain_percent_index = pg_test_array.argmin()

# plot and save the max gain figure
plt2.plot(p_test, color='red', label='prediction')
plt2.plot(y_test, color='blue', label='y_test')
plt2.plot(buy_time_max, buy_price_max, 'y+', label='buy', mew=5, ms=10)
plt2.plot(sell_time_max, sell_price_max, 'gx', label='sell', mew=5, ms=10)
plt2.ylabel('Share price')
plt2.title("{} - LSTM gain in percents: {:2}%".format(comany_name, pg_test_list[max_test_gain_percent_index]))
plt2.legend(loc='upper right')
# plt2.show()
plt2.savefig(path_save_result_images + '{}.png'.format(max_test_gain_percent_index))

print("max_test_gain_percent_index", max_test_gain_percent_index)
print("max_test_gain_percent", pg_test_array[max_test_gain_percent_index])

print("max_test_gain_percent_epsilon_1", epsilon_1_test_list[max_test_gain_percent_index])
print("max_test_gain_percent_epsilon_2", epsilon_2_test_list[max_test_gain_percent_index])

print("min_test_gain_percent_index", min_test_gain_percent_index)
print("min_test_gain_percent", pg_test_array[min_test_gain_percent_index])

print("average_test_gain_percent", np.average(pg_test_array))

loggers('max_test_gain_percent_index: {} \n'.format(max_test_gain_percent_index))
loggers('max_test_gain_percent: {} \n'.format(pg_test_array[max_test_gain_percent_index]))

loggers('max_test_gain_percent_epsilon_1: {} \n'.format(epsilon_1_test_list[max_test_gain_percent_index]))
loggers('max_test_gain_percent_epsilon_2: {} \n'.format(epsilon_2_test_list[max_test_gain_percent_index]))

loggers('min_test_gain_percent_index: {} \n'.format(min_test_gain_percent_index))
loggers('min_test_gain_percent: {} \n'.format(pg_test_array[min_test_gain_percent_index]))

loggers('average_test_gain_percent: {} \n'.format(np.average(pg_test_array)))