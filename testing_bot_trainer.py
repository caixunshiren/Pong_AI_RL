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
    params = []
    params = bt.train_bot(Xtrain, Ytrain, Rtrain, params)