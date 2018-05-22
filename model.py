#!/usr/bin/env python

import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda


class BehaviourCloning(object):
    """Class implementing bahaviour cloning."""
    def __init__(self):
        self.data_dir = 'data'
        self.images = list()
        self.measurements = list()

        self.x_train = None
        self.y_train = None

        self.model = None
        self.epochs = 1

    def load_training_data(self):
        """Loads training data and stores in in np arrays.
           Assumes data is stored in the data_dir and containd a csvfile along
           with an IMG folder containing the corresponding images.
        """
        data_paths = [p for p in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, p))]
        index_path = 'driving_log.csv'

        self.images = list()
        self.measurements = list()
        for path in data_paths:
            rows = list()
            with open(os.path.join(self.data_dir, path, index_path)) as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    rows.append(row)
            for row in rows:
                img_path = os.path.join(self.data_dir, path, row[0])
                img = cv2.imread(img_path)
                steering = float(row[3])
                self.measurements.append(steering)
                if img is not None:
                    self.images.append(img)
                else:
                    raise(RuntimeError('Could not load image: {}'.format(img_path)))

    def setup_ml_sets(self):
        self.y_train = np.array(self.measurements)
        self.x_train = np.array(self.images)

    def create_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(160, 320, 3)))
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(self.x_train, self.y_train,
                  validation_split=0.2, shuffle=True,
                  epochs=self.epochs)
        self.model.save('model.h5')


def main():
    """model training"""
    bh_cloning = BehaviourCloning()
    bh_cloning.load_training_data()
    print('Loaded {} data entries.'.format(len(bh_cloning.measurements)))
    print('Loaded {} image entries with dimensions {}'.format(
        len(bh_cloning.images), bh_cloning.images[-1].shape))
    bh_cloning.setup_ml_sets()
    bh_cloning.create_model()
    print('Model created')


if __name__ == '__main__':
    main()
