import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf
import math
#from tensorflow.python.framework import ops
tf.compat.v1.disable_eager_execution()


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

def placeholders(num_features):

    A_0 = tf.placeholder(dtype = tf.float64, shape = ([num_features,None]), name = 'A_0')

    label = tf.placeholder(dtype = tf.float64, shape = ([1,None]), name = 'label')

    reward = tf.placeholder(dtype = tf.float64, shape = ([1,None]), name = 'reward')

    return A_0,label,reward

def initialiseParameters(params):

    W1 = tf.Variable(initial_value=tf.convert_to_tensor(params['W1'], np.float64), dtype=tf.float64, name = 'W1')

    b1 = tf.Variable(initial_value=tf.convert_to_tensor(params['b1'], np.float64), dtype=tf.float64, name = 'b1')

    W2 = tf.Variable(initial_value=tf.convert_to_tensor(params['W2'], np.float64), dtype=tf.float64, name = 'W2')

    b2 = tf.Variable(initial_value=tf.convert_to_tensor(params['b2'], np.float64), dtype=tf.float64, name = 'b2')

    W3 = tf.Variable(initial_value=tf.convert_to_tensor(params['W3'], np.float64), dtype=tf.float64, name = 'W3')

    b3 = tf.Variable(initial_value=tf.convert_to_tensor(params['b3'], np.float64), dtype=tf.float64, name = 'b3')

    print("2. Checking Parameter Shapes:")
    print({"W1":W1,"b1":b1,"W2":W2,"b2":b2, "W3":W3,"b3":b3})
    print("  ")
    print("  ")
    '''

    W1 = tf.Variable(initial_value=tf.random_normal([200,600], dtype = tf.float64) * 0.01)

    b1 = tf.Variable(initial_value=tf.zeros([200,1], dtype=tf.float64))

    W2 = tf.Variable(initial_value=tf.random_normal([1,200], dtype=tf.float64) * 0.01)

    b2 = tf.Variable(initial_value=tf.zeros([1,1], dtype=tf.float64))
    '''
    return {"W1":W1,"b1":b1,"W2":W2,"b2":b2, "W3":W3,"b3":b3}

def forward_propagation(A_0,parameters):

    Z1 = tf.matmul(parameters["W1"],A_0) + parameters["b1"]

    A1 = tf.nn.relu(Z1)

    Z2 = tf.matmul(parameters["W2"],A1) + parameters["b2"]

    A2 = tf.nn.relu(Z2)

    Z3 = tf.matmul(parameters["W3"],A2) + parameters["b3"]

    A3 = tf.sigmoid(Z3) # 1 * m

    #A2 = Z2

    return Z2

def loss(logit, label, reward, m):
    entr = label * -tf.log(logit) + (1-label) * -tf.log(1-logit)
    return -tf.reduce_sum(reward * entr)

def compute_mini_batches(X, Y, R, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_R = R[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_R)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_R = R[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_R)
        mini_batches.append(mini_batch)

    return mini_batches

def shallow_model(X,Y,R, params, learning_rate, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    #ops.reset_default_graph()
    (n_x, m) = X.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y.shape[0]

    num_features = X.shape[0]

    A_0, label, reward = placeholders(num_features)

    parameters = initialiseParameters(params)

    A2 = forward_propagation(A_0, parameters)

    cost = loss(A2, label, reward, m)
    #cost = -1*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=A2,labels=label))
    #train_net = tf.train.GradientDescentOptimizer(learning_rate).maximize(cost)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    #print("1")
    with tf.Session() as sess:

        sess.run(init)

        #print("2")
        # Do the training loop
        for epoch in range(num_epochs):
            #print("3")
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            #seed = seed + 1
            minibatches = compute_mini_batches(X, Y, R, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y, minibatch_R) = minibatch


                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={A_0: minibatch_X, label: minibatch_Y, reward: minibatch_R})
                #print("running")
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))



        # lets save the parameters in a variable
        parameters = sess.run(parameters)

        #sess.reset()

        print ("Parameters have been trained!")
        return parameters




def train_bot(Xtrain, Ytrain, Rtrain, params):
    print("---------------------------")
    print("---------------------------")
    print("Policy Training Start")
    print("*       *       *")
    '''
    NN structure:
        800 features ----> 400 nets ----> sigmoid calculates P(up)
    '''

    #hyperparameters
    gamma = 0.95
    learning_rate = 0.003

    #Data Processing
    Rtrain = convert_advantage_factor(Rtrain, gamma)
    X, Y, R = concat_training_set(Xtrain, Ytrain, Rtrain)

    #Placeholder
    #print(params)

    params = shallow_model(X,Y,R, params, learning_rate, num_epochs = 1, minibatch_size = 32, print_cost = True)
    tf.reset_default_graph()
    #print(params)
    '''
    for key in params:
        params[key] = params[key].numpy()
        print(type(params[key]))
        print(params[key])
    '''
    print("*       *       *")
    print("Policy Training End")
    print("---------------------------")
    print("---------------------------")
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

############
import json

def  test():
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

    #print(params)

    params = train_bot(Xtrain, Ytrain, Rtrain, params)


#test()