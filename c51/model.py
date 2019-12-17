# from __future__ import print_function
#
# from keras.models import Model
# from keras.layers import Convolution2D, Dense, Flatten, Input
# from keras.optimizers import Adam
#
# class Networks(object):
#
#     @staticmethod
#     def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
#         """Model Value Distribution
#         With States as inputs and output Probability Distributions for all Actions
#         """
#
#         state_input = Input(shape=(input_shape))
#         cnn_feature = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(state_input)
#         cnn_feature = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(cnn_feature)
#         cnn_feature = Convolution2D(64, 3, 3, activation='relu')(cnn_feature)
#         cnn_feature = Flatten()(cnn_feature)
#         cnn_feature = Dense(512, activation='relu')(cnn_feature)
#
#         distribution_list = []
#         for i in range(action_size):
#             distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))
#
#         model = Model(input=state_input, output=distribution_list)
#
#         adam = Adam(lr=learning_rate)
#         model.compile(loss='categorical_crossentropy',optimizer=adam)
#
#         return model

import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Model, self).__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vis_layers = nn.Sequential(
            nn.Conv2d(s_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            Flatten()
        )

    def forward(self, x):

        return x
