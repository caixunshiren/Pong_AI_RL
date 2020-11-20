import numpy as np




def convert_advantage_factor(Ytrain, gamma):
    Ytrain_modified = []
    for round in Ytrain:
        for i in range(0, len(round)):
            round[i] = gamma**(len(round)-i)
        Ytrain_modified.append(round)

    return Ytrain_modified

def concat_training_set(Xtrain, Ytrain):
    '''
    TBC
    '''
    print(len(Xtrain))
    print(len(Xtrain[0]))
    print(len(Xtrain[0][0]))

    print(len(Ytrain))
    print(len(Ytrain[0]))

    X = np.array(Xtrain)
    Y = np.array(Ytrain)
    print(X.shape)
    print(Y.shape)
    type(X)

    return Xtrain, Ytrain

def normalization():
    pass

def train_bot(Xtrain, Ytrain, params):
    '''
    To be completed: implement forward prop and backward prop to train the params


    '''
    params = ["this is parameter 1","this is parameter 2"]

    '''
    NN structure:
        600 features ----> 100 nets ----> 20 nets ----> sigmoid P for up
    '''

    #hyperparameters
    gamma = 0.99

    #Data Processing
    Ytrain = convert_advantage_factor(Ytrain, gamma)
    Xtrain, Ytrain = concat_training_set(Xtrain, Ytrain)

    return params