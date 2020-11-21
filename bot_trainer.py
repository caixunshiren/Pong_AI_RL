import numpy as np
import tensorflow as tf



def convert_advantage_factor(Rtrain, gamma):
    Rtrain_modified = []
    for round in Rtrain:
        for i in range(0, len(round)):
            round[i] = gamma**(len(round)-i)
        Rtrain_modified.append(round)
    #Optional: normalize the reward

    return Rtrain_modified

def concat_training_set(Xtrain, Ytrain, Rtrain):

    X = []
    R = []

    for round_x, round_r in zip(Xtrain, Rtrain):
        X = X + round_x
        R = R+round_r

    X = np.array(X)
    Y = np.array([Ytrain]).T
    R = np.array([R]).T
    print(X.shape)
    print(Y.shape)
    print(R.shape)
    print(X)
    print(Y)
    print(R)

    return X, Y, R


def train_bot(Xtrain, Ytrain, Rtrain, params):
    '''
    To be completed: implement forward prop and backward prop to train the params


    '''
    params = ["this is parameter 1","this is parameter 2"]

    '''
    NN structure:
        600 features ----> 200 nets ----> sigmoid P for up
    '''

    #hyperparameters
    gamma = 0.99
    learning_rate = 0.01


    #Data Processing
    Rtrain = convert_advantage_factor(Rtrain, gamma)
    Xtrain, Rtrain = concat_training_set(Xtrain, Ytrain, Rtrain)

    return params











############ Archived ############
def normalization(X):
    '''
    Don't need it. Normalize in RLbot when collecting data

    norm_X = np.zeros(X.shape)
    #print(norm_X.shape)

    return norm_X
    '''
    pass





