import bot_trainer as bt
import json


if __name__ == '__main__':
    Xtrain = []
    Ytrain = []
    with open('Xtrain.txt') as f:
        Xtrain = json.load(f)
    with open('Ytrain.txt') as f:
        Ytrain = json.load(f)

    #print(Ytrain)
    #print(Xtrain[1:10])

    Ytrain = bt.convert_advantage_factor(Ytrain, 0.99)
    Xtrain, Ytrain = bt.concat_training_set(Xtrain, Ytrain)