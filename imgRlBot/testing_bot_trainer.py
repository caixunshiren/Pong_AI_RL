import bot_trainer as bt
import json
import numpy as np

if __name__ == '__main__':
    Xtrain = []
    Ytrain = []
    Rtrain = []
    with open('Xtrain.txt') as f:
        Xtrain = json.load(f)
    with open('Ytrain.txt') as f:
        Ytrain = json.load(f)
    with open('Rtrain.txt') as f:
        Rtrain = json.load(f)


    #print(Ytrain)
    #print(Xtrain[1])
    #Y = np.array(Ytrain)
    #print(Y)
    #print(Y.shape)

   # print(Ytrain)

    #Rtrain = bt.convert_advantage_factor(Rtrain, 0.99)
    #X, Y, R = bt.concat_training_set(Xtrain, Ytrain, Rtrain)

    #X = bt.normalization(X)
    H = 200
    D = 600

    params = {}
    params['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
    params['W2'] = np.random.randn(1,H) / np.sqrt(H) # Shape will be H
    params['b1'] = np.zeros((H,1))
    params['b2'] = np.zeros((1,1))

    #bt.initialiseParameters(params)
    #Rtrain = bt.convert_advantage_factor(Rtrain, 0.99)
    #X, Y, R = bt.concat_training_set(Xtrain, Ytrain, Rtrain)
    #minibatches = bt.compute_mini_batches(X, Y, R, 32)
    #print(minibatches[0])
    params = bt.train_bot(Xtrain, Ytrain, Rtrain, params)