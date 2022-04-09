from tensorflow.keras import optimizers, losses, activations, models, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, SimpleRNN, LSTM
import pdb

def baseline(nclass= 5, shape = 100):
    inp = Input(shape=(shape))
    dense_1 = Dense(nclass, activation=activations.softmax)(inp)
    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model