###################
# Pong + Reinforcement Learning + CNNs
###################
import numpy as np
import math
from tensorflow import keras

def concat_training_set(Xtrain, Ytrain, Rtrain):
    X = []
    R = []

    for round_x, round_r in zip(Xtrain, Rtrain):
        X = X + round_x
        R = R+round_r
    X = np.array(X).T
    Y = np.array([Ytrain])
    R = np.array([R])
    print("1. Checking Input Shapes:")
    print("X:",X.shape)
    print("Y:",Y.shape)
    print("R:",R.shape)
    print("  ")
    print("  ")
    #print(X)
    #print(Y)
    #print(R)
    return X, Y, R

def modified_jack_loss(eps_reward):
    def loss(y_true, y_pred):
        # prune pred b.c. of possible invalid nums (domain of log)
        pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.02,0.98))(y_pred)
        tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(pred)
        policy_loss=keras.layers.Multiply()([tmp_loss,eps_reward])
    return policy_loss
return loss

def make_models(input_shape):
    # assuming SCALE_FACTOR = 5 
    # (88 x 56 x 1) -> conv2d -> conv2d -> flatten -> 1x1
    # not using Sequential gives a little more flexibility
    input_layer = keras.layers.Input(shape=input_shape)
    
    #------- Can modify model pretty easily here!
    conv1_layer = keras.layers.Conv2D(4, 8, activation='relu', strides=(3,3), padding='valid', use_bias=True, )(input_layer)
    maxpool1_layer = keras.layers.MaxPool2D(pool_size=(2,2))(conv1_layer) # I'd imagine there is a lot of redundancy in the frames...
    conv2_layer = keras.layers.Conv2D(8, 4, activation='relu', strides=(1,1), padding='valid', use_bias=True, )(maxpool1_layer) # Maybe just one CNN is enough
    flatten1_layer = keras.layers.Flatten()(conv2_layer)
    #--------
    output_layer = keras.layers.Dense(1, activation="sigmoid", use_bias=True)(flatten1_layer) # To Bias or Not To Bias?
    
    reward_layer = keras.layers.Input(shape=(1,), name='reward_layer')

    run_model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    train_model = keras.Model.model(inputs=[input_layer, reward_layer], outputs=sigmoid_output) 
    
    train_model.compile(optimizer='adam', loss=modified_jack_loss(reward_layer),)
    
    return train_model, run_model
   
def train_model(train_model, img_list, action_list, rewards_list):
    # take from pongbot, np.arrays
    # Rtrain is already processed?
    rewards = np.expand_dims(rewards_list, 1)
    y_true = np.expand_dims(action_list,1)
    print("-----SHAPES-------")
    print("X:", img_list.shape)
    print("Y:", action_list.shape)
    print("R:", rewards_list.shape)
    train_model.fit(x=[img_list, rewards], y=y_true)

def convert_advantage_factor(Rtrain, gamma):
    Rtrain_modified = []
    for round in Rtrain:
        for i in range(0, len(round)):
            round[i] = gamma**(len(round)-i)
        Rtrain_modified.append(round)
    #Optional: normalize the reward
    return Rtrain_modified



