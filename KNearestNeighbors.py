'''
KNearestNeighbors.py contains all code related to the K-Nearest-Neighbors classifier.
Authors: Kyle Back (RUID: 187000266), Erin McGowan (RUID: 184004761)
'''

import Hyperparameters
import random
import math

# Global face variables for KNN
FACE_DIVISION_WIDTH_KNN = 5
FACE_DIVISION_HEIGHT_KNN = 2
FACE_K = 29

# Global digit variables for KNN
DIGIT_DIVISION_WIDTH_KNN = 7
DIGIT_DIVISION_HEIGHT_KNN = 7
DIGIT_K = 3

'''
Contains the image index, euclidean distance, and true label of an image from another image.
'''
class DistTuple:
    def __init__(self, image_index, dist, label):
        self.image_index = image_index
        self.dist = dist
        self.label = label

'''
Gets called from Main.py to execute the K-Nearest-Neighbors algorithm for classification.
'''
def execute_KNN(data, percent_train_data, isFaceData):

	# Code that performs hyperparameter tuning (Tuning already complete, so that's why it's commented)
	'''
	global FACE_DIVISION_WIDTH_KNN
	global FACE_DIVISION_HEIGHT_KNN
	global FACE_K
	global DIGIT_DIVISION_WIDTH_KNN
	global DIGIT_DIVISION_HEIGHT_KNN
	global DIGIT_K

	if (isFaceData):
		image_height = Hyperparameters.FACE_HEIGHT
		image_width = Hyperparameters.FACE_WIDTH
	else:
		image_height = Hyperparameters.DIGIT_HEIGHT
		image_width = Hyperparameters.DIGIT_WIDTH

	highest_accuracy = 0
	best_division_width = 0
	best_division_height = 0
	best_k = 0

	# Tunes the parameters to find the best combination
	for division_height in range(1, image_height):
		for division_width in range(1, image_width):
			for k in range(1, 30):

				# Ensures division_height is a factor of FACE_HEIGHT
				if (image_height % division_height != 0):
					continue
				
				# Ensures division_width is a factor of FACE_WIDTH
				if (image_height % division_width != 0):
					continue
				
				# Ensures k is odd
				if (k % 2 == 0):
					continue

				if (isFaceData):
					FACE_DIVISION_HEIGHT_KNN = division_height
					FACE_DIVISION_WIDTH_KNN = division_width
					FACE_K = k
				else:
					DIGIT_DIVISION_HEIGHT_KNN = division_height
					DIGIT_DIVISION_WIDTH_KNN = division_width
					DIGIT_K = k

				# Trains/tests with current combination of hyperparameters
				feature_values = train(data, isFaceData)
				accuracy = test(feature_values, data, isFaceData)

				print(str(division_height) + ', ' + str(division_width) + ', ' + str(k) + ': ' + str(round(accuracy, 3)))

				# Updates best accuracy/parameters if needed
				if (accuracy > highest_accuracy):
					highest_accuracy = accuracy
					best_division_height = division_height
					best_division_width = division_width
					best_k = k

	print('Highest accuracy of ' + str(round(highest_accuracy, 3)) + ' obtained with:')
	print('	   Division height of ' + str(best_division_height))
	print('	   Division width of ' + str(best_division_width))
	print('	   K of ' + str(best_k))
	'''

	# Randomly samples from training data set
	train_data_sample = sample_data(data.train, percent_train_data)

	# Calculates feature values, then tests
	feature_values = train(train_data_sample, isFaceData)
	accuracy = test(feature_values, data, train_data_sample, isFaceData)
	return accuracy, len(train_data_sample)

'''
Determines the feature values of the train data for K-Nearest-Neighbors algorithm.
'''
def train(train_data_sample, isFaceData):

	if (isFaceData):
		feature_values = calculate_feature_values_face(train_data_sample)
	else:
		feature_values = calculate_feature_values_digit(train_data_sample)

	return feature_values

'''
Tests the K-Nearest-Neighbors algorithm for accuracy.
'''
def test(feature_values, data, train_data_sample, isFaceData):

	correct_predictions = 0

	for image_index in range(len(data.test)):

		image = data.test[image_index]
		
		# Calculates euclidian distances, and prediction for current image
		dist_list = calculate_distances(train_data_sample, image, feature_values, isFaceData)
		dist_list.sort(key = lambda dist_tuple: dist_tuple.dist)

		if (isFaceData):
			prediction = get_prediction_face(dist_list)
		else:
			prediction = get_prediction_digit(dist_list)

		# Increments correct predictions if label is same
		if (prediction == int(data.test[image_index].label)):
			correct_predictions += 1
	
	return correct_predictions / len(data.test)

'''
Calculates the feature values for a given face data set. The features being used are:
	1. Total number of '#' characters in the image.
	2. The maximum number of '#' characters in a single division.
	3. The loaction of the maximum division. This is done by enumerating each division
	   so that each one has a unique number assigned to it. 
Returns a list of tuples that contain the feature values. 
'''
def calculate_feature_values_face(train_data_sample):

	# Declare global variables
	global FACE_DIVISION_WIDTH_KNN 
	global FACE_DIVISION_HEIGHT_KNN 

	feature_values = []

	for image in train_data_sample:

		feature_tuple = []
		total_pound_count = 0
		max_pound_count = 0
		max_count_location = 0
		curr_location = 0

		# Iterates through each of the divisions
		for start_row in range(0, Hyperparameters.FACE_HEIGHT, FACE_DIVISION_HEIGHT_KNN):
			for start_col in range(0, Hyperparameters.FACE_WIDTH, FACE_DIVISION_WIDTH_KNN):
				
				curr_location += 1
				pound_count = 0

				# Sums up the number of non-space characters in this division
				for row in range(start_row, start_row + FACE_DIVISION_HEIGHT_KNN):
					for col in range(start_col, start_col + FACE_DIVISION_WIDTH_KNN):
						
						if (image.image[row][col] == '#'):
							total_pound_count += 1
							pound_count += 1

				# Updates max pound count if needed
				if (pound_count > max_pound_count):
					max_pound_count = pound_count
					max_count_location = curr_location

		curr_location = 0
		
		feature_values.append([total_pound_count, max_pound_count, max_count_location])

	return feature_values

'''
Calculates the feature values for a given digit data set. The features being used are:
	1. Number of pixels in each division. The number of features will be equal to
	   the number of divisions in the image.
Returns a list of tuples that contain the feature values. 
'''
def calculate_feature_values_digit(train_data_sample):

	# Declare global variables
	global DIGIT_DIVISION_WIDTH_KNN 
	global DIGIT_DIVISION_HEIGHT_KNN 

	feature_values = []

	for image in train_data_sample:

		feature_tuple = []

		# Iterates through each of the divisions
		for start_row in range(0, Hyperparameters.DIGIT_HEIGHT, Hyperparameters.DIGIT_DIVISION_HEIGHT):
			for start_col in range(0, Hyperparameters.DIGIT_WIDTH, Hyperparameters.DIGIT_DIVISION_WIDTH):

				pixel_count = 0

				# Sums up the number of non-space characters in this division
				for row in range(start_row, start_row + Hyperparameters.DIGIT_DIVISION_HEIGHT):
					for col in range(start_col, start_col + Hyperparameters.DIGIT_DIVISION_WIDTH):
						if (image.image[row][col] != ' '):
							pixel_count += 1
				
				# Add value to feature tuple
				feature_tuple.append(pixel_count)

		feature_values.append(feature_tuple)

	return feature_values

'''
Calculates the distance of a specfied image to the rest of the images in a 
data set. Returns a list of DistTuple's. 
'''
def calculate_distances(train_data_sample, image, feature_values, isFaceData):

	dist_list = []

	# Calculates features of current test image depending on type
	if (isFaceData):
		image_features_A = calculate_feature_values_face([image])[0]
	else:
		image_features_A = calculate_feature_values_digit([image])[0]

	# Iterates through the feature values of training images
	for image_index in range(len(feature_values)):

		image_features_B = feature_values[image_index]
		sum = 0

		# Iterates through each feature and sums the squares of the differences
		for i in range(len(image_features_A)):
			difference = image_features_B[i] - image_features_A[i]
			sum += math.pow(difference, 2)
		
		# Calculates the distance by 
		dist = math.sqrt(sum)

		# Saves true label of image
		true_label = train_data_sample[image_index].label

		dist_list.append(DistTuple(image_index, dist, true_label))

	return dist_list

'''
Given a list of distances, takes the mode of the first K entries to calculate the predicted
label of a face image. Returns the prediction.
'''
def get_prediction_face(dist_list):

	global FACE_K

	face_count = 0

	# Iteractes through first K entries
	for i in range(FACE_K):
		
		# Increments face count if label is face
		if (dist_list[i].label == '1'):
			face_count += 1

	# Returns 1 if mode is 1, otherwise return 0
	if (face_count > math.floor(FACE_K / 2)):
		return 1
	else:
		return 0

'''
Given a list of distances, takes the mode of the first K entries to calculate the predicted
label of a digit image. Returns the prediction.
'''
def get_prediction_digit(dist_list):

	global DIGIT_K

	correct_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	max_count = 0
	max_index = 0

	# Iteractes through first K entries
	for i in range(DIGIT_K):
		
		# Increments count of digit that aligns with label
		correct_counts[int(dist_list[i].label)] += 1

		curr_count = correct_counts[int(dist_list[i].label)]

		# Update max count if needed
		if (curr_count > max_count):
			max_count = curr_count
			max_index = int(dist_list[i].label)

	return max_index

'''
Given a dataset and a percentage, returns a random sample from the
dataset with size proportonial to the pertentage. 
'''
def sample_data(data_set, percent):

    # Calculates sample size, then returns sample
    sample_size = math.floor(len(data_set) * percent)
    train_data_sample = random.sample(data_set, sample_size)
    return train_data_sample
