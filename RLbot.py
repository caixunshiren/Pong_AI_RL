import numpy as np
import copy

frame = 1
cur_side = 'right'
cur_reward = 0
last_score = [0,0]
reset = False

#frame_info
'''
bx
by

ppx1
ppy1
ppy2

pox1
poy1
poy2

+ previous frame info
'''
frame_1_info = []
frame_info = []
reward_info = []

#training
Xtrain = []
Ytrain = []
Rtrain = []

def store_frame_info_more_frames(paddle_frect, other_paddle_frect, ball_frect, n_f, table_size):
    global frame
    #global frame_1_info
    global frame_info
    global cur_reward
    global reward_info

    #update reward
    reward_info.append(cur_reward)

    #update_frame_info
    cur_frame_info = []
    if frame == 1:
        for i in range(n_f):
            cur_frame_info.append(ball_frect.pos[0] / table_size[0] - 0.5)
            cur_frame_info.append(ball_frect.pos[1] / table_size[1] - 0.5)
            cur_frame_info.append(paddle_frect.pos[0] / table_size[0] - 0.5)
            cur_frame_info.append(paddle_frect.pos[1] / table_size[1] - 0.5)
            cur_frame_info.append((paddle_frect.pos[1]+70) / table_size[1] - 0.5)
            cur_frame_info.append(other_paddle_frect.pos[0] / table_size[0] - 0.5)
            cur_frame_info.append(other_paddle_frect.pos[1] / table_size[1] - 0.5)
            cur_frame_info.append((other_paddle_frect.pos[1]+70) / table_size[1] - 0.5)
        #print(len(cur_frame_info))
        #print(cur_frame_info)
        #print("----------")
    else:
        cur_frame_info.append(ball_frect.pos[0] / table_size[0] - 0.5)
        cur_frame_info.append(ball_frect.pos[1] / table_size[1] - 0.5)
        cur_frame_info.append(paddle_frect.pos[0] / table_size[0] - 0.5)
        cur_frame_info.append(paddle_frect.pos[1] / table_size[1] - 0.5)
        cur_frame_info.append((paddle_frect.pos[1]+70) / table_size[1] - 0.5)
        cur_frame_info.append(other_paddle_frect.pos[0] / table_size[0] - 0.5)
        cur_frame_info.append(other_paddle_frect.pos[1] / table_size[1] - 0.5)
        cur_frame_info.append((other_paddle_frect.pos[1]+70) / table_size[1] - 0.5)
        cur_frame_info = cur_frame_info + copy.deepcopy(frame_info[frame-2][0:-8])

    #print(len(cur_frame_info))
    #print(cur_frame_info)
    #print("----------")
    frame_info.append(cur_frame_info)

    '''
    if cur_reward != 0:
        print(len(frame_info))
        print(len(reward_info))
        print(reward_info)
        print("------------------------")
    '''

    return

def check_side(paddle_frect, score):
    global cur_side
    global last_score

    if paddle_frect.pos[0] < 100:
        side = 'left'
    else:
        side = 'right'

    if side != cur_side or score == [0,0]:
        cur_side = side
        last_score = [0,0]

def update_reward(score):
    global cur_reward
    global reset
    global cur_side
    global last_score

    if score[0] == last_score[0] and score[1] == last_score[1]:
        reward = 0
        #print(last_score)
    else:
        #print(score)
        #print(last_score)
        #print(cur_side)
        reset = True
        if cur_side == 'left':
            cur_reward = score[0] - last_score[0] + last_score[1] - score[1]
        else:
            cur_reward = score[1] - last_score[1] + last_score[0] - score[0]
    #if cur_reward < 0:
    #    cur_reward * 0.75

def reset_round():
    global frame
    global cur_reward
    global frame_1_info
    global frame_info
    global reward_info

    frame = 1
    cur_reward = 0
    frame_1_info = []
    frame_info = []
    reward_info = []

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def forward_prop(x):
    global params
    x = np.array([x]).T
    A1 = np.dot(params['W1'], x) + params['b1'] # (H x D) . (D x 1) = (H x 1) (200 x 1)
    A1[A1<0] = 0 # ReLU introduces non-linearity

    A2 = np.dot(params['W2'], A1) + params['b2'] # (H x D) . (D x 1) = (H x 1) (20 x 1)
    A2[A2<0] = 0 # ReLU introduces non-linearity

    A3 = np.dot(params['W3'], A2) + params['b3']# This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
    A3 = sigmoid(A3)  # squashes output to  between 0 & 1 range
    return A3

#This is the main function
def pongbot(paddle_frect, other_paddle_frect, ball_frect, table_size, score = []):
    global last_score
    global reset
    global frame
    global frame_info
    global reward_info
    global Xtrain
    global Ytrain
    global Rtrain

    #print(score, last_score)

    check_side(paddle_frect, score)
    update_reward(score)
    #store_frame_info(paddle_frect, other_paddle_frect, ball_frect)
    store_frame_info_more_frames(paddle_frect, other_paddle_frect, ball_frect, 4, table_size)





    #decision making by AI
    '''
    if paddle_frect.pos[1]+paddle_frect.size[1]/2 < ball_frect.pos[1]+ball_frect.size[1]/2:
     return "down"
    else:
     return "up"
    '''
    action_prob = forward_prop(frame_info[-1])
    #print(action_prob)
    ret = 'up' if np.random.uniform() < action_prob else 'down'
    y = 1 if ret == 'up' else 0
    Ytrain.append(y)

    #end, update global variables
    frame += 1
    last_score = copy.deepcopy(score)

    if reset:
        #pass info to training
        Xtrain.append(frame_info)
        Rtrain.append(reward_info)
        #reset
        reset_round()
        reset = False

    return ret







import json
import bot_trainer as bt

#training section
def train():
    global Xtrain
    global Ytrain
    global Rtrain
    global params
    print("---------------------------")
    print("Training Data Collected!")
    #print(Rtrain)
    params = bt.train_bot(Xtrain, Ytrain, Rtrain, params)
    Xtrain = []
    Ytrain = []
    Rtrain = []
    print("Parameters Updated Successfully!")

def save_params():
    global params
    for key in params:
        params[key] = params[key].tolist()
        #print(type(params[key]))
        #print(params[key])

    filename = 'params2lsmall9995.txt'

    with open(filename, 'w') as f:
        f.write(json.dumps(params))

    print("Parameters Saved!", filename)

def save_training_sets():
    global Xtrain
    global Ytrain
    global Rtrain
    with open('Xtrain.txt', 'w') as f:
        f.write(json.dumps(Xtrain))
    with open('Ytrain.txt', 'w') as f:
        f.write(json.dumps(Ytrain))
    with open('Rtrain.txt', 'w') as f:
        f.write(json.dumps(Rtrain))



########### The Weights ############

H1 = 200
H2 = 200
D = 32

mode = 'load'
params = {}

if mode == 'new':

    params['W1'] = np.random.randn(H1,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
    params['W2'] = np.random.randn(H2,H1) / np.sqrt(H1) # Shape will be H
    params['W3'] = np.random.randn(1,H2) / np.sqrt(H2)
    params['b1'] = np.zeros((H1,1))
    params['b2'] = np.zeros((H2,1))
    params['b3'] = np.zeros((1,1))

    init_params = copy.deepcopy(params)

    for key in params:
        init_params[key] = init_params[key].tolist()

    with open('initial_params.txt', 'w') as f:
        f.write(json.dumps(init_params))

elif mode == 'load':

    with open('params2lsmall9995.txt', 'r') as f:
        params = json.load(f)

    for key in params:
        params[key] = np.array(params[key])
######################### Archived Code ########################################
'''
def store_frame_info(paddle_frect, other_paddle_frect, ball_frect):
    global frame
    global frame_1_info
    global frame_info
    global cur_reward
    global reward_info

    #get frame 1 and 2 info
    if frame == 1:
        frame_1_info.append(ball_frect.pos[0])
        frame_1_info.append(ball_frect.pos[1])
        frame_1_info.append(paddle_frect.pos[0])
        frame_1_info.append(paddle_frect.pos[1])
        frame_1_info.append(paddle_frect.pos[1]+70)
        frame_1_info.append(other_paddle_frect.pos[0])
        frame_1_info.append(other_paddle_frect.pos[1])
        frame_1_info.append(other_paddle_frect.pos[1]+70)


    if frame == 2:
        cur_frame_info = []
        cur_frame_info.append(ball_frect.pos[0])
        cur_frame_info.append(ball_frect.pos[1])
        cur_frame_info.append(paddle_frect.pos[0])
        cur_frame_info.append(paddle_frect.pos[1])
        cur_frame_info.append(paddle_frect.pos[1]+70)
        cur_frame_info.append(other_paddle_frect.pos[0])
        cur_frame_info.append(other_paddle_frect.pos[1])
        cur_frame_info.append(other_paddle_frect.pos[1]+70)
        cur_frame_info = cur_frame_info + copy.deepcopy(frame_1_info)
        frame_info.append(cur_frame_info)

        reward_info.append(cur_reward)

    #for each frame after frame 2
    if frame > 2:
        cur_frame_info = []
        cur_frame_info.append(ball_frect.pos[0])
        cur_frame_info.append(ball_frect.pos[1])
        cur_frame_info.append(paddle_frect.pos[0])
        cur_frame_info.append(paddle_frect.pos[1])
        cur_frame_info.append(paddle_frect.pos[1]+70)
        cur_frame_info.append(other_paddle_frect.pos[0])
        cur_frame_info.append(other_paddle_frect.pos[1])
        cur_frame_info.append(other_paddle_frect.pos[1]+70)
        cur_frame_info = cur_frame_info + copy.deepcopy(frame_info[frame-3][0:8])
        frame_info.append(cur_frame_info)

        reward_info.append(cur_reward)

    #for testing
    #if frame == 20:
    #    print(frame_info)

    if cur_reward != 0:
        print(len(frame_info))
        print(len(reward_info))
        print(reward_info)
        print("------------------------")

    return


'''













