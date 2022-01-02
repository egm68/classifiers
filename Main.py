'''
Main.py is the driver file for the final project.
Authors: Kyle Back (RUID: 187000266), Erin McGowan (RUID: 184004761)
'''

import KNearestNeighbors
import Hyperparameters
import Perceptron
import NaiveBayes
import statistics
import time

'''
Contains the data and labels for the training, validation, and test datasets. 
'''
class Data:
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

'''
Represents an image and its corresponding label. This is used for both faces and digits.
'''
class ImageLabelPair:
    def __init__(self, image, label):
        self.image = image
        self.label = label

'''
Extracts the face or digit data from a specfied text file.
Returns the data in the form of a single three-dimensional matrix where
the first index corresponds to the image number, the second index corresponds
to the row number of an image, and the third index corresponds to the column 
number of an image.
'''
def extractData(isFaceData, file_name):

    # Initializes height/width
    height = 0
    width = 0

    if (isFaceData): 
        # Extracting face data
        height = Hyperparameters.FACE_HEIGHT
        width = Hyperparameters.FACE_WIDTH
    else: 
        # Extracting digit data 
        height = Hyperparameters.DIGIT_HEIGHT
        width = Hyperparameters.DIGIT_WIDTH

    # Gets total number lines in the file
    total_num_lines = sum(1 for line in open(file_name, 'r'))
    num_images = int(total_num_lines / height)
    
    # Initializes matrix where extracted data will be stored
    data = [[[' ' for col in range(width)] for row in range(height)] for image_num in range(num_images)]

    # Reads in the lines from the file
    lines = open(file_name, 'r').readlines()
    lines_reverse = []

    # Reverses the list because we'll be popping items off the end (This preserves the correct order)
    for i in range(len(lines)):
        lines_reverse.append(lines.pop())

    # Extracts the data and inserts it into the matrix
    for image_num in range(num_images):
        for row in range(height):
            line = lines_reverse.pop()
            for col in range(width):
                data[image_num][row][col] = line[col]

    return data

'''
Reads in the labels from a specfied file.
Returns an array containing all of them. 
'''
def extractLabels(file_name):
    labels = []
    
    # Reads in the lines from the file
    lines = open(file_name, 'r').readlines()

    # Iterates through the lines and saves the labels
    for line in lines:
        labels.append(line[0])

    return labels

'''
Combines a list of images with a list corresponding to the images' labels
into a single list. Returns the combined list. 
'''
def combineData(data, labels):

    pairList = []

    # Iterates through the images and labels
    for image_index in range(len(data)):
        pair = ImageLabelPair(data[image_index], labels[image_index])
        pairList.append(pair)

    return pairList

'''
Calculates the standard deviation and mean for training on
a given proportion of a data set with Perceptron. Must indicate whether face 
data or digit data is supplied. Returns the stand deviation and mean.
'''
def get_mean_std_perceptron(data, percent_train_data, isFaceData):

    # Initializes variables
    accuracies = []
    weights = []
    bias = 0

    # Trains for several iterations, and saves the test accuracies
    for i in range(5):
        if (isFaceData):
            weights, bias, num_data_points = Perceptron.train_face(data, percent_train_data)
            accuracies.append(Perceptron.test_face(data.test, weights, bias))
        else:
            weights, bias, num_data_points = Perceptron.train_digit(data, percent_train_data)
            accuracies.append(Perceptron.test_digit(data.test, weights, bias))

    # Calculates mean and standard deviation 
    mean = statistics.mean(accuracies)
    std = statistics.stdev(accuracies)

    return std, mean

'''
Calculates the standard deviation and mean for training on
a given proportion of a data set with Naive Bayes. Must indicate whether face 
data or digit data is supplied. Returns the stand deviation and mean.
'''
def get_mean_std_naive_bayes(data, percent_train_data, isFaceData):

    # Initializes variables
    accuracies = []

    # Trains for several iterations, and saves the test accuracies
    for i in range(5):
        if (isFaceData):
            p_y_face, p_y_notFace, y_face, y_notFace, num_data_points = NaiveBayes.train_face(data, percent_train_data)
            accuracy = NaiveBayes.test_face(data.test, p_y_face, p_y_notFace, y_face, y_notFace)
            accuracies.append(accuracy)
        else:
            p_y_0, p_y_1, p_y_2, p_y_3, p_y_4, p_y_5, p_y_6, p_y_7, p_y_8, p_y_9, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, num_data_points = NaiveBayes.train_digit(data, percent_train_data)
            accuracy = NaiveBayes.test_digit(data.test, p_y_0, p_y_1, p_y_2, p_y_3, p_y_4, p_y_5, p_y_6, p_y_7, p_y_8, p_y_9, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9)
            accuracies.append(accuracy)

    # Calculates mean and standard deviation 
    mean = statistics.mean(accuracies)
    std = statistics.stdev(accuracies)

    return std, mean

'''
Calculates the standard deviation and mean for training on
a given proportion of a data set with K-Nearest_Neighbors. Must indicate whether face 
data or digit data is supplied. Returns the stand deviation and mean.
'''
def get_mean_std_KNN(data, percent_train_data, isFaceData):

    # Initializes variables
    accuracies = []

    # Trains for several iterations, and saves the test accuracies
    for i in range(5):
        accuracy, num_data_points = KNearestNeighbors.execute_KNN(data, percent_train_data, isFaceData)
        accuracies.append(accuracy)

    # Calculates mean and standard deviation 
    mean = statistics.mean(accuracies)
    std = statistics.stdev(accuracies)

    return std, mean

'''
Prints out the test accuracies, mean accuracy, and standard deviation of a given classifier.
Iterates through different percentages of the training data ranging from 10% to 100%.
The parameter, classifierType, indicates the type of classifier being reported ('Perceptron',
'Naive Bayes', or 'K-Nearest-Neighbors').
'''
def print_accuracy(data, isFaceData, classifierType):

    dataType = 'face' if isFaceData else 'digit'

    print('\033[95m')
    print('\nTraining and testing ' + str(classifierType) + ' for ' + str(dataType) + ' classification:')
    print('\033[0m')

    # Trains classifier on subsets of the train data ranging from 10% to 100%
    for i in range(1, 11):

        percent_train_data = round(i * 0.1, 1)

        if (classifierType == 'Perceptron'):

            if (isFaceData):
                # Training Perceptron on face data
                start_time = time.time()
                weights, bias, num_data_points = Perceptron.train_face(data, percent_train_data)
                end_time = time.time()
                training_time = end_time - start_time

                # Calculates the test accuracy 
                accuracy = Perceptron.test_face(data.test, weights, bias)

                # Calculates the standard deviation and mean
                std, mean = get_mean_std_perceptron(data, percent_train_data, True)

            else:
                # Training Perceptron on digit data
                start_time = time.time()
                weights, bias, num_data_points = Perceptron.train_digit(data, percent_train_data)
                end_time = time.time()
                training_time = end_time - start_time

                # Calculates the test accuracy 
                accuracy = Perceptron.test_digit(data.test, weights, bias)

                # Calculates the standard deviation and mean
                std, mean = get_mean_std_perceptron(data, percent_train_data, False)

        elif (classifierType == 'Naive Bayes'):

            if (isFaceData):
                # Training Naive Bayes on face data
                start_time = time.time()
                p_y_face, p_y_notFace, y_face, y_notFace, num_data_points = NaiveBayes.train_face(data, percent_train_data)
                end_time = time.time()
                training_time = end_time - start_time

                # Calculates the test accuracy 
                accuracy = NaiveBayes.test_face(data.test, p_y_face, p_y_notFace, y_face, y_notFace)

                # Calculates the standard deviation and mean
                std, mean = get_mean_std_naive_bayes(data, percent_train_data, True)

            else:
                # Training Naive Bayes on digit data
                start_time = time.time()
                p_y_0, p_y_1, p_y_2, p_y_3, p_y_4, p_y_5, p_y_6, p_y_7, p_y_8, p_y_9, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, num_data_points = NaiveBayes.train_digit(data, percent_train_data)
                end_time = time.time()
                training_time = end_time - start_time

                # Calculates the test accuracy 
                accuracy = NaiveBayes.test_digit(data.test, p_y_0, p_y_1, p_y_2, p_y_3, p_y_4, p_y_5, p_y_6, p_y_7, p_y_8, p_y_9, y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9)

                # Calculates the standard deviation and mean
                std, mean = get_mean_std_naive_bayes(data, percent_train_data, False)

        else:

            if (isFaceData):
                # Training K-Nearest-Neighbors on face data
                start_time = time.time()
                accuracy, num_data_points = KNearestNeighbors.execute_KNN(data, percent_train_data, True)
                end_time = time.time()
                training_time = end_time - start_time

                # Calculates the standard deviation and mean
                std, mean = get_mean_std_KNN(data, percent_train_data, True)

            else:
                # Training K-Nearest-Neighbors on digit data
                start_time = time.time()
                accuracy, num_data_points = KNearestNeighbors.execute_KNN(data, percent_train_data, False)
                end_time = time.time()
                training_time = end_time - start_time

                # Calculates the standard deviation and mean
                std, mean = get_mean_std_KNN(data, percent_train_data, False)

        print('   Training on ' + str(int(percent_train_data * 100)) + '% of the data:')
        print('      Training time took ' + str(round(training_time, 2)) + ' seconds.')
        print('      Test accuracy is ' + str(round(accuracy * 100, 3)) + '%.')
        print('      Mean is ' + str(round(mean * 100, 3)) + '% for ' + str(num_data_points) + ' data points.')
        print('      Standard deviation is ' + str(round(std * 100, 3)) + '% for ' + str(num_data_points) + ' data points.')

# Extracts face data and labels
faceTrainingData = extractData(True, 'data/facedata/facedatatrain')
faceTrainingLabels = extractLabels('data/facedata/facedatatrainlabels')
faceTrainData = combineData(faceTrainingData, faceTrainingLabels)
faceValidationData = extractData(True, 'data/facedata/facedatavalidation')
faceValidationLabels = extractLabels('data/facedata/facedatavalidationlabels')
faceValidationData = combineData(faceValidationData, faceValidationLabels)
faceTestData = extractData(True, 'data/facedata/facedatatest')
faceTestLabels = extractLabels('data/facedata/facedatatestlabels')
faceTestData = combineData(faceTestData, faceTestLabels)

faceData = Data(faceTrainData, faceValidationData, faceTestData)

# Extracts digit data and labels
digitTrainingData = extractData(False, 'data/digitdata/trainingimages')
digitTrainingLabels = extractLabels('data/digitdata/traininglabels')
digitTrainData = combineData(digitTrainingData, digitTrainingLabels)
digitValidationData = extractData(False, 'data/digitdata/validationimages')
digitValidationLabels = extractLabels('data/digitdata/validationlabels')
digitValidationData = combineData(digitValidationData, digitValidationLabels)
digitTestData = extractData(False, 'data/digitdata/testimages')
digitTestLabels = extractLabels('data/digitdata/testlabels')
digitTestData = combineData(digitTestData, digitTestLabels)

digitData = Data(digitTrainData, digitValidationData, digitTestData)

# Trains, tests, and prints the metrics for each classifier for face and digit data
print_accuracy(faceData, True, 'Perceptron')
print_accuracy(digitData, False, 'Perceptron')
print_accuracy(faceData, True, 'Naive Bayes')
print_accuracy(digitData, False, 'Naive Bayes')
print_accuracy(faceData, True, 'K-Nearest-Neighbors')
print_accuracy(digitData, False, 'K-Nearest-Neighbors')





