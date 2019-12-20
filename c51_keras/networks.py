#!/usr/bin/env python
from __future__ import print_function

from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam


class Networks(object):

    @staticmethod
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution
        With States as inputs and output Probability Distributions for all Actions
        """

        state_input = Input(shape=(input_shape))
        cnn_feature = Conv2D(32, (8, 8), activation='relu')(state_input)
        cnn_feature = Conv2D(64, (4, 4), activation='relu')(cnn_feature)
        cnn_feature = Conv2D(64, (3, 3), activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='relu')(cnn_feature)

        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))

        model = Model(inputs=state_input, outputs=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model