import pandas as pd
import matplotlib.pyplot as plt

# matplotlib.inline
import numpy as np
import matplotlib.pylab
matplotlib.pylab.rcParams['figure.figsize'] = 20, 10
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("TSLA(1).csv")
df.head()


df["Date"] = pd.to_datetime(df.Date, format = "%Y-%m-%d") # defining a column

df.index = df['Date']

plt.figure(figsize = (16,8))
plt.plot(df["Close"], label = 'Close Price History')

data = df.sort_index(ascending = True, axis = 0)
new_dataset = pd.DataFrame(index = range(0,len(df)), columns = ['Date', 'Close'])

for i in range(0,len(data)):
   new_dataset["Date"][i] = data['Date'][i]
   new_dataset["Close"][i] = data['Close'][i]



scaler=MinMaxScaler(feature_range = (0,1))
final_dataset = new_dataset.values

train_data = final_dataset[0:987]
test_data = final_dataset[987:]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date")





scaler = MinMaxScaler(feature_range(0,1))
scaled_data = scaler.fit_transform(train_data)

x_train_data,y_train_data = [],[]

for i in range(60,len(train_data)):
   x_train_data.append(scaled_data[i-60:i,0])
   y_train_data.append(scaled_data[i,0])

   # need to figure out this weird indexing thing

x_train_data,y_train_data = np.array(x_train_data),np.array(y_train_data)
x_train_data = np.reshape(x_train_data, x_train_data.shape[0],x_train_data.shape[1])


#Training the LSTM Model

model = Sequential()

model.add(LSTM(50, input_shape = (x_train_data[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

inputs_data = new_dataset[len(new_dataset)-len(test_data)- 60:]
inputs_data = inputs_data.reshape(-1,1)
inputs_data = scaler.transform(inputs_data)

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train_data,y_train_data, epochs = 1, batch_size = 1 , verbose = 2)

#Test data from the new dataset split into train and test data

x_test = []
for i in range(60,inputs_data.shape[0]):
   x_test.append(inputs_data[i-60:i,0]) # don't understand this indexing
x_test = np.array(x_test)

x_test = np.reshape(x_test, x_test.shape[0],1)
predicted_closing_price = model.predict(x_test)

model.save("saved lstm model")

#predict the stock trends

train_data = new_dataset[:987]
test_data = new_dataset[987:]

test_data['Predictions'] = predicted_closing_price
plt.plot(train_data["Close"])
plt.plot(test_data[['Close','Predictions']])





















