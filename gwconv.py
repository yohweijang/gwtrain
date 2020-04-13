import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense, Conv1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from conv1dTP import conv1dTP

NTYPE = 10
RATE = 8192
A = 1
N = 30
batch_size=200
epochs = 2000

gwtrain = pd.read_csv('GW_train_full.csv', index_col=0)
gw_test = pd.read_csv('GW_test_full.csv' , index_col=0) 

gwtrain = np.array(gwtrain)
gw_test = np.array(gw_test)

xtrain = gwtrain.reshape((-1, RATE, 1))
xtest1 = gw_test.reshape((-1, RATE, 1))

trd0 = xtrain.shape[0]
ted0 = xtest1.shape[0]
print(trd0,ted0)
trbig = np.max(xtrain,axis=1)
tebig = np.max(xtest1,axis=1)

for i in range(trd0): xtrain[i,:,0] = xtrain[i,:,0]/trbig[i,0]
for i in range(ted0): xtest1[i,:,0] = xtest1[i,:,0]/tebig[i,0]

tot = 1200
trt = tot - trd0

xtraid = xtest1[:trt]
xtest1 = xtest1[trt:]
xtrain = np.concatenate((xtrain, xtraid), axis=0)

kernel_size = 16
enfilters = [64,64,32,32,32]
defilters = [32,32,64,64]
midense = 512
latent_dim = 32

### encoder
inputs = Input(shape=(RATE, 1))
x = inputs

for filters in enfilters:
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=2,
               activation='tanh', padding='same')(x)

x_shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(midense)(x)

xoutputs = Dense(latent_dim)(x)

xencoder = Model(inputs, xoutputs)
xencoder.summary()
#plot_model(xencoder, to_file='xencoder.png', show_shapes=True)

### decoder
xlatent = Input(shape=latent_dim)

x = Dense(midense)(xlatent)
x = Dense(x_shape[1] * x_shape[2])(x)
x = Reshape((x_shape[1], x_shape[2]))(x)

for filters in defilters:
    x = conv1dTP(filters=filters, kernel_size=kernel_size, strides=2,
                 activation='tanh', padding='same')(x)

xoutputs = conv1dTP(filters=1, kernel_size=kernel_size, strides=2,
                    activation='tanh', padding='same')(x)

xdecoder = Model(xlatent, xoutputs)
xdecoder.summary()
#plot_model(xdecoder, to_file='xdecoder.png', show_shapes=True)

### autoencoder
xautoencoder = Model(inputs, xdecoder(xencoder(inputs)))
xautoencoder.summary()
#plot_model(xautoencoder, to_file='xautoencoder.png', show_shapes=True)

xautoencoder.compile(loss='mse', optimizer='adam')
hist = xautoencoder.fit(xtrain, xtrain, batch_size=batch_size, epochs=epochs,
                        shuffle=True, validation_data=(xtest1, xtest1))

ypred = xautoencoder.predict(xtest1)

plt.figure()
plt.plot(xtest1[1,:,0], color='blue')
plt.plot(ypred[ 1,:,0], color='red' )
plt.title("epoch: %s" %(epochs))
#plt.show()
plt.savefig('gwmatch1.pdf')

plt.figure()
plt.plot(xtest1[100,:,0], color='blue')
plt.plot(ypred[ 100,:,0], color='red' )
plt.title("epoch: %s" %(epochs))
#plt.show()
plt.savefig('gwmatch2.pdf')

# list all data in history
print(hist.history.keys())

# summarize history for loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
#plt.show()
plt.savefig('loss.pdf')