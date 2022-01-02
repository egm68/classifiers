'''
Hyperparameters.py contains all the hyperparameters used accross Main.py, Perceptron.py, and NaiveBayes.py.
KNearestNeighbors.py has its own set of hyperparameters located in its file so tuning can be performed.
Authors: Kyle Back (RUID: 187000266), Erin McGowan (RUID: 184004761)
'''

FACE_WIDTH = 60
FACE_HEIGHT = 70
FACE_DIVISION_WIDTH = 6
FACE_DIVISION_HEIGHT = 7
DIGIT_WIDTH = 28
DIGIT_HEIGHT = 28
DIGIT_DIVISION_WIDTH = 4
DIGIT_DIVISION_HEIGHT = 4
NUM_FEATURES_FACE = int(FACE_WIDTH / FACE_DIVISION_WIDTH) * int(FACE_HEIGHT / FACE_DIVISION_HEIGHT)
NUM_FEATURES_DIGIT = int(DIGIT_WIDTH / DIGIT_DIVISION_WIDTH) * int(DIGIT_HEIGHT / DIGIT_DIVISION_HEIGHT)