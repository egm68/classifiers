'''
Perceptron.py contains all code related training, validating, and testing the Perceptron classifier.
Authors: Kyle Back (RUID: 187000266), Erin McGowan (RUID: 184004761)
'''

import Hyperparameters
import random
import math

'''
Trains the model on the face data by repeatedly updating the weights
associated with each feature, and the bias.
'''
def train_face(data, percent_train_data):
    
    # Initalizes parameters
    weights = []
    bias = 0

    # Initalizes weights to zero
    for i in range(Hyperparameters.NUM_FEATURES_FACE):
        weights.append(0)

    # Randomly samples from training data set
    train_data_sample = sample_data(data.train, percent_train_data)
    
    # Use test data
    for i in range(10):
        for image_index in range(len(train_data_sample)):
            weights, bias = adjust_weights_face(train_data_sample, image_index, weights, bias)
    
    return weights, bias, len(train_data_sample)
    

'''
Gets called on each indivisual image to calculate the feature values for faces.
Weights and bias get adjusted accordingly at the end, and validation set is checked for accuracy.
Returns the updated list of weights and the bias.
'''
def adjust_weights_face(train_data_sample, image_index, weights, bias):

    feature_values = extract_features_face(train_data_sample[image_index].image)
    function_value = calculate_function_value_face(feature_values, weights, bias)
    
    if (function_value >= 0 and train_data_sample[image_index].label == '0'):
        # Incorrectly predicted a face: Substract from weights and bias
        for i in range(len(weights)):
            weights[i] -= feature_values[i]
        bias -= 1

    elif (function_value < 0 and train_data_sample[image_index].label == '1'):
        # Incorrectly predicted not a face: Add to weights and bias
        for i in range(len(weights)):
            weights[i] += feature_values[i]
        bias += 1

    #accuracy = check_validation_face(faceValidationData, faceValidationLabels, weights, bias)
    #print('Epoch ' + str(image_index) + ': Validation acurracy is ' + str(round(accuracy, 3)))
    
    return weights, bias

'''
Calculates the values for each feature of a face image.
Returns a list of feature values.
'''
def extract_features_face(image):
    
    feature_values = []

    # Feature 1: Number of pound characters in each division
    total_pound_count = 0

    # Iterates through each of the divisions
    for start_row in range(0, Hyperparameters.FACE_HEIGHT, Hyperparameters.FACE_DIVISION_HEIGHT):
        for start_col in range(0, Hyperparameters.FACE_WIDTH, Hyperparameters.FACE_DIVISION_WIDTH):

            pound_count = 0

            # Sums up the number of pound characters in this division
            for row in range(start_row, start_row + Hyperparameters.FACE_DIVISION_HEIGHT):
                for col in range(start_col, start_col + Hyperparameters.FACE_DIVISION_WIDTH):
                    if (image[row][col] == '#'):
                        total_pound_count += 1
                        pound_count += 1
            
            feature_values.append(pound_count)


    return feature_values
    
'''
Calculates the f(x) value given the feature values and the weights associated
with them, and the bias value. Returns the f(x) value.
'''
def calculate_function_value_face(feature_values, weights, bias):
    function_value = 0

    # Sums the product of the feature values and weights
    for i in range(len(feature_values)):
        function_value += feature_values[i] * weights[i]

    # Adds the bias to value
    function_value += bias
    return function_value

'''
Goes through each image in the validation set for faces and determines the validation accuracy 
with the current weights for each feature value. Returns the validation accuracy as
a decimal with three points of precision. 
'''
def test_face(data, weights, bias):

    correct_prediction_count = 0

    # Iterates through all validation images and calculates feature values
    for image_index in range(len(data)):
        feature_values = extract_features_face(data[image_index].image)
        function_value = calculate_function_value_face(feature_values, weights, bias)

        if ((function_value >= 0 and data[image_index].label == '1') or (function_value < 0 and data[image_index].label == '0')):
            correct_prediction_count += 1
    
    return correct_prediction_count / len(data)

'''
Trains the model on the digit data by repeatedly updating the weights
associated with each feature, and the bias.
'''
def train_digit(data, percent_train_data):
    
    # Initalizes parameters (Set of weights and bias for each digit)
    weights = [[], [], [], [], [], [], [], [], [], []]
    bias = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Initializes weights to zero
    for digit_num in range(len(weights)):
        for i in range(Hyperparameters.NUM_FEATURES_DIGIT):
            weights[digit_num].append(0)

    # Randomly samples from training data set
    train_data_sample = sample_data(data.train, percent_train_data)

    # Train the classifer
    for i in range(10):
        for image_index in range(len(train_data_sample)):
            weights, bias = adjust_weights_digit(train_data_sample, image_index, weights, bias)
    
    return weights, bias, len(train_data_sample)

'''
Gets called on each indivisual image to calculate the feature values for digits.
Weights and bias get adjusted accordingly at the end, and validation set is checked for accuracy.
Returns the updated list of weights and the bias.
'''
def adjust_weights_digit(data, image_index, weights, bias):

    feature_values = extract_features_digit(data[image_index].image)
    predicted_digit = calculate_function_value_digit(feature_values, weights, bias)
    true_digit = int(data[image_index].label)

    # Adjust weights
    if (true_digit != predicted_digit):

        # Increase weights for true digit
        for i in range(len(weights[true_digit])):
            weights[true_digit][i] += feature_values[i]
        bias[true_digit] += 1

        # Decrease weights for predicted digit
        for i in range(len(weights[predicted_digit])):
            weights[predicted_digit][i] -= feature_values[i]
        bias[predicted_digit] -= 1
    
    return weights, bias

'''
Calculates the values for each feature of a digit image.
Returns a list of feature values.
'''
def extract_features_digit(image):

    # Feature 1: Number of pound characters in each division
    feature_values = []

    # Iterates through each of the divisions
    for start_row in range(0, Hyperparameters.DIGIT_HEIGHT, Hyperparameters.DIGIT_DIVISION_HEIGHT):
        for start_col in range(0, Hyperparameters.DIGIT_WIDTH, Hyperparameters.DIGIT_DIVISION_WIDTH):

            pound_count = 0

            # Sums up the number of pound characters in this division
            for row in range(start_row, start_row + Hyperparameters.DIGIT_DIVISION_HEIGHT):
                for col in range(start_col, start_col + Hyperparameters.DIGIT_DIVISION_WIDTH):
                    if (image[row][col] == '#'):
                        pound_count += 1
            
            feature_values.append(pound_count)

    return feature_values

'''
Calculates all of the f(x) values for each of the digits, and returns
the maximum value along with the index of the digit.
'''
def calculate_function_value_digit(feature_values, weights, bias):
    
    max_function_value = 0
    max_function_index = 0
    
    # Iterates through each weight subset
    for digit_index in range(len(weights)):

        function_value = 0

        # Iterates through each value in a weight subset
        for i in range(Hyperparameters.NUM_FEATURES_DIGIT):
            function_value += feature_values[i] * weights[digit_index][i]

        function_value += bias[digit_index]

        # Checks if new max value was found
        if (function_value > max_function_value):
            max_function_value = function_value
            max_function_index = digit_index
    
    return max_function_index

'''
Goes through each image in the validation set for digits and determines the validation accuracy 
with the current weights for each feature value. Returns the validation accuracy as
a decimal with three points of precision. 
'''
def test_digit(data, weights, bias):

    correct_prediction_count = 0

    # Iterates through all validation images and calculates feature values
    for image_index in range(len(data)):
        feature_values = extract_features_digit(data[image_index].image)
        predicted_digit = calculate_function_value_digit(feature_values, weights, bias)
        true_digit = int(data[image_index].label)

        if (predicted_digit == true_digit):
            correct_prediction_count += 1
    
    return correct_prediction_count / len(data)

'''
Given a dataset and a percentage, returns a random sample from the
dataset with size proportonial to the pertentage. 
'''
def sample_data(data_set, percent):

    # Calculates sample size, then returns sample
    sample_size = math.floor(len(data_set) * percent)
    train_data_sample = random.sample(data_set, sample_size)
    return train_data_sample

'''
Given an image (2-Dimensional array), prints the contents.
'''
def print_image(image):

    for row in range(len(image)):
        for col in range(len(image[0])):
            print(image[row][col], end='')
        print()
    
