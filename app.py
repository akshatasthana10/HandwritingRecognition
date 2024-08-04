import copy
import math
import cv2 as cv
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential


def image_method(h):
	xx = ""
	# time.sleep(1)
	raw_image = cv.imread(h,0)
	imageHeight = raw_image.shape[0]
	imageWidth = raw_image.shape[1]
	MedianBlurredImage = cv.medianBlur(raw_image,3)
	InvertedImage = 255 - MedianBlurredImage
	kernel = np.ones((3,3), np.uint8)
	DilatedImage0 = cv.dilate(InvertedImage, kernel, iterations=1)
	kernel = np.ones((2,2), np.uint8)
	DilatedImage = cv.dilate(DilatedImage0, kernel, iterations=1)
	GaussianBlurredImage = cv.GaussianBlur(DilatedImage, (3, 3), 0)  # smoothing or blurring
	ret3, OTImage = cv.threshold(GaussianBlurredImage, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	ProcessedImage = copy.copy(OTImage)
	contours, hierarchy = cv.findContours(ProcessedImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	numberofChars = len(contours)
	a = []
	average_char_height = 0
	average_char_width = 0

	for cnt in contours:
		x,y,w,h = cv.boundingRect(cnt)
		a.append([int(y+h/2),int(x+w/2),y,x,y+h,x+w,-1])
		average_char_height = average_char_height + h
		average_char_width = average_char_width + w

	average_char_height = average_char_height/numberofChars
	average_char_width = average_char_width/numberofChars
	fullStopSize = int(math.sqrt(average_char_height * average_char_height / 4))

	print(average_char_height, average_char_width, fullStopSize)

	def secondSort(val):
		return val[0]

	a.sort(key=secondSort)

	heightFromTop = 0
	width = int(average_char_height*2/3)
	row = 0
	flag = 0
	rowlist = []
	shift = int(width/5)
	max_scan = imageHeight-average_char_height-2*shift-10

	print("width =",width,", shift =",shift,", max_scan =",max_scan)

	while heightFromTop < max_scan:
		while (flag == 0 and heightFromTop < max_scan):
			for el in a:
				if (el[0] > heightFromTop and el[0] < (heightFromTop + width)):
					flag = 1
					break
			heightFromTop = heightFromTop + shift
		while flag == 1 and heightFromTop < max_scan:
			flag = 0
			for el in a:
				if (el[0] > heightFromTop and el[0] < (heightFromTop + width)):
					flag = 1
					break
			heightFromTop = heightFromTop + shift
		row = row + 1
		flag = 0
		rowlist.append(heightFromTop)
	rowlist.pop()
	row = row - 1

	i = 0
	j = 0
	lengthOfa = len(a)
	lengthOfRowList = len(rowlist)
	while (i < lengthOfRowList):
		while (j < lengthOfa):
			if (a[j][0] < rowlist[i]):
				a[j][6] = i
				j = j + 1
			else:
				i = i + 1
				break
		if (j == lengthOfa):
			i = i + 1

	a = sorted(a, key=lambda x: (x[6], x[1]))
	print(a)
	aCopy = a.copy()

	def checkDot(x, fullStopSize):
		if (x[4] - x[2] < fullStopSize and x[5] - x[3] < fullStopSize):
			return True
		else:
			return False

	def DirectlyAbovePrevious(x, y):
		if (x[6] != y[6]):
			return False
		elif (x[0] - y[0] > average_char_height / 3 and y[1] - x[1] < 4 * average_char_width / 5):
			return True

	def DirectlyAboveNext(y, x):
		if (x[6] != y[6]):
			return False
		elif (x[0] - y[0] > average_char_height / 3 and x[1] - y[1] < 4 * average_char_width / 5):
			return True

	i = 0
	a.clear()

	while (i < len(aCopy)):
		# print(i)
		if (checkDot(aCopy[i], fullStopSize)):
			# print("Case", i)
			if (i>0 and DirectlyAbovePrevious(aCopy[i - 1], aCopy[i])):
				# print("caseA")
				newTop = aCopy[i][2]
				newBottom = max(aCopy[i][4], aCopy[i - 1][4])
				newLeft = min(aCopy[i][3], aCopy[i - 1][3])
				newRight = max(aCopy[i][5], aCopy[i - 1][5])
				newHeight = newBottom - newTop
				newWidth = newRight - newLeft
				newX = int(newLeft + newWidth / 2)
				newY = int(newTop + newHeight / 2)

				a.pop(len(a) - 1)
				a.append([newY, newX, newTop, newLeft, newBottom, newRight, aCopy[i][6]])
				i = i + 1

			elif (i<len(aCopy)-1 and DirectlyAboveNext(aCopy[i], aCopy[i + 1])):
				# print("caseB")
				newTop = min(aCopy[i][2], aCopy[i + 1][2])
				newBottom = max(aCopy[i][4], aCopy[i + 1][4])
				newLeft = min(aCopy[i][3], aCopy[i + 1][3])
				newRight = max(aCopy[i][5], aCopy[i + 1][5])
				newHeight = newBottom - newTop
				newWidth = newRight - newLeft
				newX = int(newLeft + newWidth / 2)
				newY = int(newTop + newHeight / 2)

				a.append([newY, newX, newTop, newLeft, newBottom, newRight, aCopy[i][6]])
				i = i + 2

			else:
				a.append(aCopy[i])
				i = i + 1
		else:
			a.append(aCopy[i])
			i = i + 1

	print(a)
	print(aCopy)
	print(len(a), len(aCopy))
	numberofChars = len(a)


	kernel2 = np.ones((2, 2), np.uint8)
	ProcessedImage = cv.erode(ProcessedImage, kernel2, iterations=2)
	imagelist = []

	for el in a:
		im3 = ProcessedImage[el[2]:el[4], el[3]:el[5]] # we will get the image inside the bounding box
		im4 = np.zeros((28, 28))
		if (el[4] - el[2]) / (el[5] - el[3]) > 2.2: # l i
			im3 = cv.resize(im3, (8, 22))
			im4[3:25, 10:18] = im3
		else:
			if (el[4] - el[2]) / (el[5] - el[3]) > 1.2: # g h k
				im3 = cv.resize(im3, (12, 22))
				im4[3:25, 8:20] = im3
			else:
				im3 = cv.resize(im3, (18, 22)) # a m w n
				im4[3:25, 5:23] = im3
		im4 = cv.dilate(im4, kernel2, iterations=1)  # dilation followed by erosion to remove Background noise
		imagelist.append(im4)

	seed = 7
	np.random.seed(seed)
	num_classes = 47
	def larger_model():
		model = Sequential()
		model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(15, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(num_classes, activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # harshita

		return model
	# build the model
	model = larger_model()
	model.load_weights(r".\AZaz09weights")
	output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
	'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
	'f', 'g', 'h', 'n', 'q', 'r', 't']

	imageArray = np.asarray(imagelist)
	imageArray = imageArray / 255
	flippedImageList = []
	rows, cols = imageArray[0].shape
	M = cv.getRotationMatrix2D((cols / 2, rows / 2), 270, 1) # defining rotation matrix

	for image in imageArray:
		dst = cv.warpAffine(image, M, (cols, rows))  # rotating the character images
		flippedImageList.append(cv.flip(dst, 1))  # flipping the character images
	flippedImageArray = np.asarray(flippedImageList)
	predictedClasses = model.predict_classes(flippedImageArray.reshape(imageArray.shape[0], 28, 28, 1))  #prdeiction

	for i in range(0, predictedClasses.shape[0]):
		if predictedClasses[i] == 0:
			predictedClasses[i] = 24

	def checkDot(x, fullStopSize):
		if (x[4] - x[2] < fullStopSize and x[5] - x[3] < fullStopSize):
			return True
		else:
			return False

	def checkSpace(x, y, characterWidth):
		if ((y[3] - x[5]) > int(characterWidth) or x[6] != y[6]):
			return True
		else:
			return False

	def checkFullStop(x, y, characterHeight):
		if ((x[2] + x[4]) / 2 - (y[2] + y[4]) / 2 < characterHeight / 3):
			return True
		else:
			return False

	def checkNewLine(x, y):
		if y[6] - x[6] == 1:
			return True
		else:
			return False

	def capToSmall(x):
		if x.isupper():
			return x.lower()
		else:
			return x


	xx=xx+output_labels[predictedClasses[0]]
	if checkSpace(a[0], a[1], average_char_width):
		xx = xx + ' '

	fullStop = 0
	i = 1
	while (i < numberofChars):
		if (i<numberofChars-1 and checkDot(a[i+1],fullStopSize)):
			if(i<numberofChars-1 and checkFullStop(a[i],a[i+1],average_char_height)):
				xx=xx+capToSmall(output_labels[predictedClasses[i]])
				xx=xx+'. '
				if (i < numberofChars-2 and checkNewLine(a[i + 1], a[i + 2])):
					xx=xx+'\n'
				i = i + 2
				fullStop = 1
				continue
		if i < numberofChars-1 and checkNewLine(a[i], a[i + 1]):
			xx = xx+capToSmall(output_labels[predictedClasses[i]])+'\n'
			i = i + 1
			continue
		if fullStop:
			xx = xx + output_labels[predictedClasses[i]]
			fullStop = 0
		else:
			xx=xx+capToSmall(output_labels[predictedClasses[i]])
		if(i<numberofChars-1 and checkSpace(a[i],a[i+1],average_char_width)):
			xx=xx+" "
		i = i + 1

	return xx
