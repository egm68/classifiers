'''
NaiveBayes.py contains all code related training, validating, and testing the Naive Bayes classifier.
Authors: Kyle Back (RUID: 187000266), Erin McGowan (RUID: 184004761)
'''

import Hyperparameters
import random
import math
import numpy


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
Trains the model on the face data by populating a probablity table
associated with features from each training image.
'''
def train_face(data, percent_train_data):

    # Randomly samples from training data set
    train_data_sample = sample_data(data.train, percent_train_data)
    
    # Initalizes tables and vars
    y_rows, y_cols = (Hyperparameters.NUM_FEATURES_FACE, (Hyperparameters.FACE_DIVISION_WIDTH * Hyperparameters.FACE_DIVISION_HEIGHT) + 1)
    y_face = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_notFace = [[0 for i in range(y_cols)] for j in range(y_rows)]
    #rows = images, columns = division, a11 would be the number of # in division 1 of image 1
    faceFeatureCounts = []
    notFaceFeatureCounts = []
    faceCount = 0
    notFaceCount = 0

    # Iterates through training data and calculates number of face images, number of not face images,
    for i in range(len(train_data_sample)):
        image = train_data_sample[i].image
        imageLabel = train_data_sample[i].label

        if (int(imageLabel) == 1):
            faceCount = faceCount + 1
            feature_values = extract_features_face(image)
            faceFeatureCounts.append(feature_values)

        elif (int(imageLabel) == 0):
            notFaceCount = notFaceCount + 1
            feature_values = extract_features_face(image)
            notFaceFeatureCounts.append(feature_values)
    
    tfaceFeatureCounts = numpy.transpose(faceFeatureCounts)
    tnotFaceFeatureCounts = numpy.transpose(notFaceFeatureCounts)
    
    #populate y_face
    for row in range(0, Hyperparameters.NUM_FEATURES_FACE):
        for col in range(0, (Hyperparameters.FACE_DIVISION_WIDTH * Hyperparameters.FACE_DIVISION_HEIGHT)):
            imCount = image_count(tfaceFeatureCounts, row, col, faceCount)
            y_face[row][col] = (imCount / faceCount)
    
    #populate y_notFace
    for row in range(0, Hyperparameters.NUM_FEATURES_FACE):
        for col in range(0, (Hyperparameters.FACE_DIVISION_WIDTH * Hyperparameters.FACE_DIVISION_HEIGHT)):
            imCount = image_count(tnotFaceFeatureCounts, row, col, notFaceCount)
            y_notFace[row][col] = (imCount / notFaceCount)
    
    # Priors
    #images with face / total images
    p_y_face = faceCount / len(train_data_sample)
    #images withOUT face / total images
    p_y_notFace = notFaceCount / len(train_data_sample)
    
    return p_y_face, p_y_notFace, y_face, y_notFace, len(train_data_sample) 


'''
Goes through feature counts for each image to find the number of images where a given feature
contains a given number of # (to populate y_face and y_notFace)
'''
def image_count(featureCountsTable, featureNumber, poundCount, faceOrNotCount):
    imCount = 0
    for i in range(faceOrNotCount):
        if featureCountsTable[featureNumber][i] == poundCount:
            imCount = imCount + 1 
    return imCount  


'''
Calculates and returns P(y = face | X).
'''
def calculate_face_prob(feature_values, y_face, p_y_face):
    face_prob = 1
    
    for feature in range(len(feature_values)):
        feature_col = feature_values[feature]
        feature_prob = y_face[feature][feature_col]
        if feature_prob != 0:
            face_prob = face_prob * feature_prob
        else: 
            face_prob = face_prob * 0.001
    face_prob = face_prob * p_y_face

    return face_prob


'''
Calculates and returns P(y = NOT face | X).
'''
def calculate_NOT_face_prob(feature_values, y_notFace, p_y_notFace):
    NOT_face_prob = 1
    
    for feature in range(len(feature_values)):
        feature_prob = y_notFace[feature][feature_values[feature]]
        if feature_prob != 0:
            NOT_face_prob = NOT_face_prob * feature_prob
        else:
            NOT_face_prob = NOT_face_prob * 0.001
    
    NOT_face_prob = NOT_face_prob * p_y_notFace
    
    return NOT_face_prob

'''
Goes through each image in the test set for faces and determines the test accuracy. 
Returns the test accuracy as a decimal with three points of precision. 
'''
def test_face(data, p_y_face, p_y_notFace, y_face, y_notFace):

    correct_prediction_count = 0

    # Iterates through all validation images and calculates feature values
    for image_index in range(len(data)):
        feature_values = extract_features_face(data[image_index].image)
        face_prob = calculate_face_prob(feature_values, y_face, p_y_face)
        NOT_face_prob = calculate_NOT_face_prob(feature_values, y_notFace, p_y_notFace)

        if ((face_prob >= NOT_face_prob and data[image_index].label == '1') or (face_prob < NOT_face_prob and data[image_index].label == '0')):
            correct_prediction_count += 1
    
    return correct_prediction_count / len(data)
    

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
Trains the model on the digit data by populating a probablity table
associated with features from each training image.
'''
def train_digit(data, percent_train_data):

    # Randomly samples from training data set
    train_data_sample = sample_data(data.train, percent_train_data)
    
    # Initalizes tables and vars 
    y_rows, y_cols = (Hyperparameters.NUM_FEATURES_DIGIT, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT) + 1)
    y_0 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_1 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_2 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_3 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_4 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_5 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_6 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_7 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_8 = [[0 for i in range(y_cols)] for j in range(y_rows)]
    y_9 = [[0 for i in range(y_cols)] for j in range(y_rows)]

    #rows = images, columns = division, a11 would be the number of # in division 1 of image 1
    FeatureCounts_0 = []
    Count_0 = 0
    FeatureCounts_1 = []
    Count_1 = 0
    FeatureCounts_2 = []
    Count_2 = 0
    FeatureCounts_3 = []
    Count_3 = 0
    FeatureCounts_4 = []
    Count_4 = 0
    FeatureCounts_5 = []
    Count_5 = 0
    FeatureCounts_6 = []
    Count_6 = 0
    FeatureCounts_7 = []
    Count_7 = 0
    FeatureCounts_8 = []
    Count_8 = 0
    FeatureCounts_9 = []
    Count_9 = 0

    # Iterates through training data and calculates number of digit images, number of not digit images,
    for i in range(len(train_data_sample)):
        image = train_data_sample[i].image
        imageLabel = train_data_sample[i].label

        if (int(imageLabel) == 0):
            Count_0 = Count_0 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_0.append(feature_values)

        elif (int(imageLabel) == 1):
            Count_1 = Count_1 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_1.append(feature_values)

        elif (int(imageLabel) == 2):
            Count_2 = Count_2 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_2.append(feature_values)

        elif (int(imageLabel) == 3):
            Count_3 = Count_3 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_3.append(feature_values)

        elif (int(imageLabel) == 4):
            Count_4 = Count_4 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_4.append(feature_values)

        elif (int(imageLabel) == 5):
            Count_5 = Count_5 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_5.append(feature_values)

        elif (int(imageLabel) == 6):
            Count_6 = Count_6 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_6.append(feature_values)

        elif (int(imageLabel) == 7):
            Count_7 = Count_7 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_7.append(feature_values)

        elif (int(imageLabel) == 8):
            Count_8 = Count_8 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_8.append(feature_values)

        elif (int(imageLabel) == 9):
            Count_9 = Count_9 + 1
            feature_values = extract_features_digit(image)
            FeatureCounts_9.append(feature_values)
    
    tFeatureCounts_0 = numpy.transpose(FeatureCounts_0)
    tFeatureCounts_1 = numpy.transpose(FeatureCounts_1)
    tFeatureCounts_2 = numpy.transpose(FeatureCounts_2)
    tFeatureCounts_3 = numpy.transpose(FeatureCounts_3)
    tFeatureCounts_4 = numpy.transpose(FeatureCounts_4)
    tFeatureCounts_5 = numpy.transpose(FeatureCounts_5)
    tFeatureCounts_6 = numpy.transpose(FeatureCounts_6)
    tFeatureCounts_7 = numpy.transpose(FeatureCounts_7)
    tFeatureCounts_8 = numpy.transpose(FeatureCounts_8)
    tFeatureCounts_9 = numpy.transpose(FeatureCounts_9)
    
    #populate y_0
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_0, row, col, Count_0)
            y_0[row][col] = (imCount / Count_0)

    #populate y_1
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_1, row, col, Count_1)
            y_1[row][col] = (imCount / Count_1)

    #populate y_2
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_2, row, col, Count_2)
            y_2[row][col] = (imCount / Count_2)

    #populate y_3
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_3, row, col, Count_3)
            y_3[row][col] = (imCount / Count_3)

    #populate y_4
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_4, row, col, Count_4)
            y_4[row][col] = (imCount / Count_4)

    #populate y_5
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_5, row, col, Count_5)
            y_5[row][col] = (imCount / Count_5)

    #populate y_6
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_6, row, col, Count_6)
            y_6[row][col] = (imCount / Count_6)

    #populate y_7
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_7, row, col, Count_7)
            y_7[row][col] = (imCount / Count_7)

    #populate y_8
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_8, row, col, Count_8)
            y_8[row][col] = (imCount / Count_8)

    #populate y_9
    for row in range(0, Hyperparameters.NUM_FEATURES_DIGIT):
        for col in range(0, (Hyperparameters.DIGIT_DIVISION_WIDTH * Hyperparameters.DIGIT_DIVISION_HEIGHT)):
            imCount = image_count(tFeatureCounts_9, row, col, Count_9)
            y_9[row][col] = (imCount / Count_9)
    
    # Priors
    p_y_0 = Count_0 / len(train_data_sample)
    p_y_1 = Count_1 / len(train_data_sample)
    p_y_2 = Count_2 / len(train_data_sample)
    p_y_3 = Count_3 / len(train_data_sample)
    p_y_4 = Count_4 / len(train_data_sample)
    p_y_5 = Count_5 / len(train_data_sample)
    p_y_6 = Count_6 / len(train_data_sample)
    p_y_7 = Count_7 / len(train_data_sample)
    p_y_8 = Count_8 / len(train_data_sample)
    p_y_9 = Count_9 / len(train_data_sample)

    
    return p_y_0, p_y_1, p_y_2, p_y_3, p_y_4, p_y_5, p_y_6, p_y_7, p_y_8, p_y_9, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, len(train_data_sample) 


'''
Calculates and returns P(y = digit | X).
'''
def calculate_digit_prob(feature_values, y_digit, p_y_digit):
    digit_prob = 1

    #print("start")
    for feature in range(len(feature_values)):
        feature_col = feature_values[feature]
        feature_prob = y_digit[feature][feature_col]
        if feature_prob != 0:
            digit_prob = digit_prob * feature_prob
        else: 
            digit_prob = digit_prob * 0.001
    digit_prob = digit_prob * p_y_digit
    #print("end")

    return digit_prob


'''
Goes through each image in the test set for digits and determines the test accuracy. 
Returns the test accuracy as a decimal with three points of precision. 
'''
def test_digit(data, p_y_0, p_y_1, p_y_2, p_y_3, p_y_4, p_y_5, p_y_6, p_y_7, p_y_8, p_y_9, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9):

    correct_prediction_count = 0

    # Iterates through all validation images and calculates feature values
    for image_index in range(len(data)):
        feature_values = extract_features_digit(data[image_index].image)

        prob_0 = calculate_digit_prob(feature_values, y_0, p_y_0)
        prob_1 = calculate_digit_prob(feature_values, y_1, p_y_1)
        prob_2 = calculate_digit_prob(feature_values, y_2, p_y_2)
        prob_3 = calculate_digit_prob(feature_values, y_3, p_y_3)
        prob_4 = calculate_digit_prob(feature_values, y_4, p_y_4)
        prob_5 = calculate_digit_prob(feature_values, y_5, p_y_5)
        prob_6 = calculate_digit_prob(feature_values, y_6, p_y_6)
        prob_7 = calculate_digit_prob(feature_values, y_7, p_y_7)
        prob_8 = calculate_digit_prob(feature_values, y_8, p_y_8)
        prob_9 = calculate_digit_prob(feature_values, y_9, p_y_9)

        arr = [prob_0, prob_1, prob_2, prob_3, prob_4, prob_5, prob_6, prob_7, prob_8, prob_9]

        pred = arr.index(max(arr))

        if ((pred == 0 and data[image_index].label == '0') or (pred == 1 and data[image_index].label == '1') or (pred == 2 and data[image_index].label == '2') or (pred == 3 and data[image_index].label == '3') or (pred == 4 and data[image_index].label == '4') or (pred == 5 and data[image_index].label == '5') or (pred == 6 and data[image_index].label == '6') or (pred == 7 and data[image_index].label == '7') or (pred == 8 and data[image_index].label == '8') or (pred == 9 and data[image_index].label == '9')):
            correct_prediction_count += 1
    
    return correct_prediction_count / len(data)
      