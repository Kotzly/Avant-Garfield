import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

class AvantModel(nn.Module):
    
    def __init__(self, use_embedding=True):
        super(AvantModel, self).__init__()

       	if USE_EMBEDDING:
            # x_in = Input(shape=x_shape[1:])
            # x = Dense(PARAM_SIZE_COMIC, use_bias=False, kernel_initializer=RandomNormal(stddev=1e-4))(x_in)
            # x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
        else:
            x_in = Input(shape=y_shape[1:])
            x = TimeDistributed(Conv2D(40, (5,5), strides=(2,2), padding='same'))(x_in)
            x = LeakyReLU(0.2)(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            x = TimeDistributed(Conv2D(80, (5,5), strides=(2,2), padding='same'))(x)
            x = LeakyReLU(0.2)(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            x = TimeDistributed(Conv2D(120, (5,5), strides=(2,2), padding='same'))(x)
            x = LeakyReLU(0.2)(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)

            x = TimeDistributed(Conv2D(160, (5,5), strides=(2,2), padding='same'))(x)
            x = LeakyReLU(0.2)(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            x = TimeDistributed(Conv2D(200, (5,5), strides=(2,2), padding='same'))(x)
            x = LeakyReLU(0.2)(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            x = TimeDistributed(Conv2D(200, (5,5)))(x)
            x = LeakyReLU(0.2)(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            x = Reshape((num_panels, 200*4*4))(x)

            x = TimeDistributed(Dense(PARAM_SIZE_PANEL))(x)
            x = LeakyReLU(0.2)(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)

            x = Flatten()(x)

            x = Dense(PARAM_SIZE_COMIC)(x)
            x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
 
        x = Dense(1200, name='encoder')(x)
        x = LeakyReLU(0.2)(x)
            
        x = Dense(num_panels * PARAM_SIZE_PANEL)(x)
        x = LeakyReLU(0.2)(x)
        x = Reshape((num_panels, PARAM_SIZE_PANEL))(x)
 
        x = TimeDistributed(Dense(1600))(x)
        x = LeakyReLU(0.2)(x)
        
        x = TimeDistributed(Dense(240*4*4))(x)
        x = LeakyReLU(0.2)(x)
        x = Reshape((num_panels, 240, 4, 4))(x)
        
        x = TimeDistributed(Conv2DTranspose(200, (5,5)))(x)
        x = LeakyReLU(0.2)(x)
 
        x = TimeDistributed(Conv2DTranspose(160, (5,5), strides=(2,2), padding='same'))(x)
        x = LeakyReLU(0.2)(x)

        x = TimeDistributed(Conv2DTranspose(120, (5,5), strides=(2,2), padding='same'))(x)
        x = LeakyReLU(0.2)(x)

        x = TimeDistributed(Conv2DTranspose(80, (5,5), strides=(2,2), padding='same'))(x)
        x = LeakyReLU(0.2)(x)

        x = TimeDistributed(Conv2DTranspose(40, (5,5), strides=(2,2), padding='same'))(x)
        x = LeakyReLU(0.2)(x)
        
        x = TimeDistributed(Conv2DTranspose(num_colors, (5,5), strides=(2,2), padding='same', activation='sigmoid'))(x)
        
        model = Model(x_in, x)
        model.compile(optimizer=Adam(lr=LR), loss='mse')
