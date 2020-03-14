from keras.models import Model
from keras.models import Sequential
from keras import layers
from keras import initializers
from keras import backend as K
from keras.layers import Layer
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, InputLayer, Softmax, Lambda, ZeroPadding2D
from keras.models import Sequential, Model
from keras import backend as K

from skimage.transform import resize
from skimage.io import imread
from skimage import color

from os.path import join as pjoin
from matplotlib import pyplot as plt
import scipy.ndimage.interpolation as sni
import numpy as np
#import caffe
import h5py
import keras
import h5py

class CustomBN(Layer):

    def __init__(self, axis = -1, moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', **kwargs):
        super(CustomBN, self).__init__(**kwargs)
        self.axis = axis
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (initializers.get(moving_variance_initializer))
        self.scale_factor = 999.98236
        self.eps = 1e-5

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        dim = input_shape[self.axis]
        shape = (dim,)

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=True)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=True)

        super(CustomBN, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return ( x - self.moving_mean / self.scale_factor ) / K.sqrt(
            self.moving_variance / self.scale_factor + self.eps
        )

    def get_config(self):
        config = {
            "axis" : self.axis,
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer)
        }
        base_config = super(CustomBN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def get_model(filename, pretrained = True):

    #creazione della sequenza keras per il modello Convolutional Neural Network
    model = Sequential()

    model.add(InputLayer(input_shape=(224,224,1)))

    #conv1
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(64, (3,3), activation='relu', name='bw_conv1_1'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(64, (3,3), activation='relu', strides=2, name='conv1_2'))
    model.add(CustomBN(name = "conv1_2norm"))
    #model.add(BatchNormalization(scale = False, center = False, moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv1_2norm'))
    #model.add(BatchNormalization(moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv1_2norm'))

    #conv2
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(128, (3,3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(128, (3,3), activation='relu', strides=2, name='conv2_2'))
    model.add(CustomBN(name = "conv2_2norm"))
    #model.add(BatchNormalization(scale = False, center = False, moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv2_2norm'))

    #conv3
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(256, (3,3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(256, (3,3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(256, (3,3), activation='relu', strides=2, name='conv3_3'))
    model.add(CustomBN(name = "conv3_3norm"))
    #model.add(BatchNormalization(scale = False, center = False, moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv3_3norm'))

    #conv4
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', strides=1, dilation_rate=1, name='conv4_1'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', strides=1, dilation_rate=1, name='conv4_2'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', strides=1, dilation_rate=1, name='conv4_3'))
    model.add(CustomBN(name = "conv4_3norm"))
    #model.add(BatchNormalization(scale = False, center = False, moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv4_3norm'))

    #conv5
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', strides=1, dilation_rate=2, name='conv5_1'))
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', strides=1, dilation_rate=2, name='conv5_2'))
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', strides=1, dilation_rate=2, name='conv5_3'))
    model.add(CustomBN(name = "conv5_3norm"))
    #model.add(BatchNormalization(scale = False, center = False, moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv5_3norm'))

    #conv6
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', dilation_rate=2, name='conv6_1'))
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', dilation_rate=2, name='conv6_2'))
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', dilation_rate=2, name='conv6_3'))
    model.add(CustomBN(name = "conv6_3norm"))
    #model.add(BatchNormalization(scale = False, center = False, moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv6_3norm'))

    #conv7
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', dilation_rate=1, name='conv7_1'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', dilation_rate=1, name='conv7_2'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(512, (3,3), activation='relu', dilation_rate=1, name='conv7_3'))
    model.add(CustomBN(name = "conv7_3norm"))
    #model.add(BatchNormalization(scale = False, center = False, moving_mean_initializer = 'zeros', moving_variance_initializer = 'zeros', name ='conv7_3norm'))

    #conv8
    model.add(Conv2DTranspose(256, (4,4), activation='relu', strides=2, dilation_rate=1, padding = "same", name='conv8_1'))

    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(256, (3,3), activation='relu', strides=1, dilation_rate=1, name='conv8_2'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format=None))
    model.add(Conv2D(256, (3,3), activation='relu', dilation_rate=1, name='conv8_3'))

    model.add(Conv2D(313, (1,1), strides=1, dilation_rate=1, name='conv8_313'))
    model.add(Lambda(lambda x: x * 2.606, name='conv8_313_rh'))
    model.add(Softmax(name='class8_313_rh'))
    model.add(Conv2D(2,(1,1),strides=1,activation='linear', padding = "same", dilation_rate=1, name='class8_ab'))

    ##Load convertited weight Layer by Layer
    if pretrained:
        f = h5py.File(filename, 'r')
        wnames = [ key for key in f.keys() ]
        print(wnames)
        gamma = 1
        beta = 0
        for layer in model.layers:
            if(layer.name in wnames):
                w = f[layer.name][layer.name]
                if('kernel:0' in w.keys() and 'bias:0' in w.keys()):
                    layer.set_weights([w['kernel:0'], w['bias:0']])
                    print("UPDATING " + layer.name + " WEIGHTS")
                elif('moving_mean:0' in w.keys() and 'moving_variance:0' in w.keys()):
                    val = w['moving_mean:0'].shape
                    layer.set_weights([w['moving_mean:0'], w['moving_variance:0']])
                    print("UPDATING " + layer.name + " WEIGHTS as BATCH NORM weight = \n{}, \n{}".format(
                        w['moving_mean:0'],
                        w['moving_variance:0']
                    ))

                else:
                    print("DISCARDING " + layer.name + ", w.keys() = " + str(w.keys()))
                    print("Expected Weights = {}".format(layer.get_weights()))

    ##Gamma and beta are the scaling and shift applied after normalization (cf. BN paper). gamma = 1, beta = 0 is "do nothing."
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.build()

    model.summary()
    return model

if __name__ == "__main__":

    data_root = pjoin("data")
    weights_fname_keras  = pjoin(data_root, "util", "colorization_release_v2_norot.h5")
    net = get_model(weights_fname_keras, True)
