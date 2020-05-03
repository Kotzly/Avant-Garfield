import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

PANEL_2D_DIMS = (300, 300)
INPUT_DIMS = PANEL_2D_DIMS[0]*PANEL_2D_DIMS[1]
PARAM_SIZE_COMIC = 800
class AvantModel(nn.Module):
    
    def __init__(self, use_embedding=True):
        super(AvantModel, self).__init__()

       	if USE_EMBEDDING:
            # x_in = Input(shape=x_shape[1:])
            # x = Dense(PARAM_SIZE_COMIC, use_bias=False, kernel_initializer=RandomNormal(stddev=1e-4))(x_in)

            _input_linear = nn.Linear(
                                INPUT_DIMS,
                                PARAM_SIZE_COMIC,
                                bias=False
                            )
            nn.init.normal_(_input_linear.weight, std=1e-4)
            input_layer = nn.Sequential(
                                    _input_linear,
                                    nn.BatchNorm1d(PARAM_SIZE_COMIC)
                                )
            # x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)

        else:
            # x_in = Input(shape=y_shape[1:])
            # x = TimeDistributed(Conv2D(40, (5,5), strides=(2,2), padding='same'))(x_in)
            # x = LeakyReLU(0.2)(x)
            # x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            conv_1 = nn.Conv2d(3, 40, (5, 5), stride=(2, 2), padding=(2, 2))
            lrelu_1 = nn.LeakyReLU(.2)
            bn_1 = nn.BatchNorm2d(40)
            layer_1 = 
                nn.Sequential(
                        conv_1,
                        lrelu_1,
                        bn_1
                )

            # x = TimeDistributed(Conv2D(80, (5,5), strides=(2,2), padding='same'))(x)
            # x = LeakyReLU(0.2)(x)
            # x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            # x = TimeDistributed(Conv2D(120, (5,5), strides=(2,2), padding='same'))(x)
            # x = LeakyReLU(0.2)(x)
            # x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)

            # x = TimeDistributed(Conv2D(160, (5,5), strides=(2,2), padding='same'))(x)
            # x = LeakyReLU(0.2)(x)
            # x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            # x = TimeDistributed(Conv2D(200, (5,5), strides=(2,2), padding='same'))(x)
            # x = LeakyReLU(0.2)(x)
            # x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            # x = TimeDistributed(Conv2D(200, (5,5)))(x)
            # x = LeakyReLU(0.2)(x)
            # x = TimeDistributed(BatchNormalization(momentum=BN_M, axis=1))(x)
            
            conv_2 = nn.Conv2d(40, 80, (5, 5), stride=(2, 2), padding=(2, 2))
            lrelu_2 = nn.LeakyReLU(.2)
            bn_2 = nn.BatchNorm2d(80)
            layer_2 = 
                nn.Sequential(
                        conv_2,
                        lrelu_2,
                        bn_2
                )

            conv_3 = nn.Conv2d(80, 120, (5, 5), stride=(2, 2), padding=(2, 2))
            lrelu_3 = nn.LeakyReLU(.2)
            bn_3 = nn.BatchNorm2d(120)
            layer_3 = 
                nn.Sequential(
                        conv_3,
                        lrelu_3,
                        bn_3
                )

            conv_4 = nn.Conv2d(120, 160, (5, 5), stride=(2, 2), padding=(2, 2))
            lrelu_4 = nn.LeakyReLU(.2)
            bn_4 = nn.BatchNorm2d(160)
            layer_4 = 
                nn.Sequential(
                        conv_4,
                        lrelu_4,
                        bn_4
                )

            conv_5 = nn.Conv2d(160, 200, (5, 5), stride=(2, 2), padding=(2, 2))
            lrelu_5 = nn.LeakyReLU(.2)
            bn_5 = nn.BatchNorm2d(200)
            layer_5 = 
                nn.Sequential(
                        conv_5,
                        lrelu_5,
                        bn_5
                )

            # x = Reshape((num_panels, 200*4*4))(x)



            # x = TimeDistributed(Dense(PARAM_SIZE_PANEL))(x)
            # x = LeakyReLU(0.2)(x)
            # x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)

            _linear_mid = nn.Linear(
                    input_layer.weight.flatten().size(),
                    400
                )
            nn.init.normal_(_input_linear.weight, std=1e-4)
            mid_layer = nn.Sequential(
                            _input_linear,
                            nn.LeakyReLU(.2),
                            nn.BatchNorm1d(400)
                        )

            # x = Flatten()(x)

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
