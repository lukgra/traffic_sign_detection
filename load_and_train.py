import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import pathlib
import json

# global variables
DATA_DIR = pathlib.Path(os.path.join('.', 'data'))
BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 32

with open('CLASSES.json', 'r') as fp:
	CLASSES = json.load(fp)

CLASS_NAMES = list(CLASSES.values())
CLASS_INDXES = list(range(len(CLASS_NAMES)))
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_COUNT = len(list(DATA_DIR.glob('Train/*/*.png')))

def transform_data(file_path):
	# get label
	class_indx = tf.strings.split(file_path, os.path.sep)[-2]
	label = tf.one_hot(tf.strings.to_number(class_indx, tf.int32), 43)
	# read image
	img = tf.io.read_file(file_path)
	# decode and resize image
	img = tf.image.decode_png(img, channels=3)
	img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
	return img, label

def configure_dataset(ds):
	ds = ds.cache()
	ds = ds.shuffle(buffer_size=1000)
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	return ds

def build_model():
	model = keras.models.Sequential([
		keras.layers.experimental.preprocessing.Rescaling(scale=1./255.),
		keras.layers.Conv2D(filters=72, kernel_size=(3, 3), strides=1,
							activation="relu",  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
		keras.layers.MaxPooling2D(pool_size=(2), strides=2),
		keras.layers.Conv2D(filters=72, kernel_size=(5, 5), strides=1, activation="relu"),
		keras.layers.MaxPooling2D(pool_size=(2), strides=2),
		keras.layers.Conv2D(filters=144, kernel_size=(5, 5), activation="relu"),
		keras.layers.MaxPooling2D(pool_size=(1)),
		keras.layers.Flatten(),
		keras.layers.Dense(units=86, activation="relu"),
		keras.layers.BatchNormalization(),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(units=43, activation="softmax"),
	])

	model.compile(
		optimizer='adam',
		loss="categorical_crossentropy",
		metrics=['accuracy']
	)
	return model

def main():
	# Initializing dataset
	init_ds = tf.data.Dataset.list_files(str(DATA_DIR/'Train/*/*'), shuffle=False)
	init_ds = init_ds.shuffle(IMAGE_COUNT, reshuffle_each_iteration=False)

	# Splitting dataset into train and validation
	val_size = int(IMAGE_COUNT * 0.2)
	train_ds = init_ds.skip(val_size)
	val_ds = init_ds.take(val_size)

	# Setting images for parallel loading and processing
	train_ds = train_ds.map(transform_data, num_parallel_calls=AUTOTUNE)
	val_ds = val_ds.map(transform_data, num_parallel_calls=AUTOTUNE)

	train_ds = configure_dataset(train_ds)
	val_ds = configure_dataset(val_ds)

	model = build_model()
	checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join('model', 'model.h5'), save_best_only=True)

	history = model.fit(
		train_ds,
		validation_data=val_ds,
		batch_size=BATCH_SIZE,
		epochs=30,
		callbacks=[checkpoint_cb]
	)

if __name__=='__main__':
	main()
