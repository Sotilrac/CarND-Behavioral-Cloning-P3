#!/usr/bin/env python

import os
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense


class BehaviourCloning(object):
	"""Class implementing bahaviour cloning."""
	def __init__(self):
		self.data_dir = 'data'
		self.images = list()
		self.measurements = list()

		self.x_train = None
		self.y_train = None

		self.model = None

	def load_training_data(self):
		"""Loads training data and stores in in np arrays.
		   Assumes data is stored in the data_dir and containd a csvfile along
		   with an IMG folder containing the corresponding images.
		"""
		data_paths = [p for p in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, p))]
		img_path = 'IMG'
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
				img_path = os.path.join(path, row[0])
				img = cv2.imread(img_path)
				steering = float(row[3])
				self.images.append(img)
				self.measurements.append(steering)

	def setup_ml_sets(self):
		self.y_train = np.array(self.measurements)
		self.x_train = np.array(self.images)
		self.x_train = self.preprocess(self.x_train)

	def preprocess(self, imgs):
		"""Crop, normalize images, not implemented"""
		return imgs

	def create_model(self):
		self.model = Sequential()
		model.add(Flatten(input_shape(160, 320, 3)))
		model.add(Dense(1))

		model.compile(loss='mse', optimizer='adam')
		model.fit(x_train, y_train,
			      validation_split=0.2, shuffle=True, nb_epoch=3)
		model.save('model.h5')


def main():
	bh_cloning = BehaviourCloning()
	bh_cloning.load_training_data()
	print(len(bh_cloning.measurements))

if __name__ == '__main__':
	main()
