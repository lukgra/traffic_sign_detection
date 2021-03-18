import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import json
from numpy import argmax
from keras.models import load_model
from tensorflow.image import decode_png, resize
from tensorflow.io import read_file

def load_image(file_path):
	img = read_file(file_path)
	# decode and resize image
	img = decode_png(img, channels=3)
	img = resize(img, [32, 32])
	return img

if __name__=='__main__':
	file_path = os.path.normpath(sys.argv[1])
	if not os.path.isfile(file_path):
		print('Invalid path to the image file')
		sys.exit()

	with open('CLASSES.json', 'r') as fp:
		CLASSES = json.load(fp)

	model = load_model(os.path.join('model', 'model.h5'))
	img = load_image(file_path)
	img = img[None,:,:,:]
	
	predicted_class = argmax(model.predict(img), axis=-1)

	print('The sign is: ', CLASSES[str(predicted_class[0])])




