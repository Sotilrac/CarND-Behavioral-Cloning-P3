#!/usr/bin/env python

import os
import sys
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D


class BehaviourCloning(object):
    """Class implementing bahaviour cloning."""
    def __init__(self, epochs):
        self.data_dir = 'data'
        self.images = list()
        self.measurements = list()

        self.x_train = None
        self.y_train = None

        self.model = None
        self.epochs = epochs

        self.turn_steer = 2.0

    def load_training_data(self):
        """Loads training data and stores in in np arrays.
           Assumes data is stored in the data_dir and containd a csvfile along
           with an IMG folder containing the corresponding images.
        """
        data_paths = [p for p in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, p))]
        index_path = 'driving_log.csv'

        self.images = list()
        self.measurements = list()
        # Get data from all data folders
        for path in data_paths:
            rows = list()
            # Read CSV files
            with open(os.path.join(self.data_dir, path, index_path)) as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    rows.append(row)
            # Use the CSV data to fetch images and get steering angle
            for row in rows:
                # Read all images. Center, left (-), right(+)
                for ind in range(3):
                    img_path = os.path.join(self.data_dir, path, row[0])
                    img = cv2.imread(img_path)
                    if img is not None:
                        if ind == 0:
                            steering = float(row[3])
                        # Apply manual steering value for L and R views
                        elif ind == 1:
                            steering = -self.turn_steer
                        else:
                            steering = self.turn_steer

                        # Store data
                        self.measurements.append(steering)
                        self.images.append(img)

                        #Flip image
                        img = np.fliplr(img)
                        # Store data
                        self.measurements.append(-steering)
                        self.images.append(img)

                    else:
                        raise(RuntimeError(
                            'Could not load image: {}'.format(img_path)))

        
    def setup_ml_sets(self):
        self.y_train = np.array(self.measurements)
        self.x_train = np.array(self.images)

    def create_model_simple(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160, 320, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(self.x_train, self.y_train,
                  validation_split=0.2, shuffle=True,
                  epochs=self.epochs)
        self.model.save('model.h5')

    def create_model_lenet(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160, 320, 3)))
        self.model.add(Cropping2D(cropping=((60,25), (0,0))))
        self.model.add(Conv2D(6, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(6, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(120))
        self.model.add(Dense(84))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(self.x_train, self.y_train,
                  validation_split=0.2, shuffle=True,
                  epochs=self.epochs)
        self.model.save('model.h5')

def main():
    """model training"""
    try:
        epochs = int(sys.argv[1])
    except IndexError:
        epochs = 2
        print('Defaulting epochs to {}'.format(epochs))

    bh_cloning = BehaviourCloning(epochs)
    bh_cloning.load_training_data()
    print('Loaded {} data entries.'.format(len(bh_cloning.measurements)))
    print('Loaded {} image entries with dimensions {}'.format(
        len(bh_cloning.images), bh_cloning.images[-1].shape))
    bh_cloning.setup_ml_sets()
    bh_cloning.create_model_lenet()
    print('Model done')


if __name__ == '__main__':
    main()
