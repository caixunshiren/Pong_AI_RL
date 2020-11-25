import numpy as np
import copy
import cv2

frame = 1
cur_side = 'left'
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
SCALE_FACTOR = 5
TABLE_SIZE = (440, 280)


def render_frame(p1_frect, p2_frect, b_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim):
    global SCALE_FACTOR
    img = np.zeros((table_size[1], table_size[0]))
    # draw in paddles in white
    top_left_x = round(p1_frect.pos[0]-half_paddle_width)
    top_left_y = round(p1_frect.pos[1]+half_paddle_height)
    bottom_right_x = round(p1_frect.pos[0] + half_paddle_width)
    bottom_right_y = round(p1_frect.pos[1] - half_paddle_height)
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 1, -1)

    top_left_x = round(p2_frect.pos[0]-half_paddle_width)
    top_left_y = round(p2_frect.pos[1]+half_paddle_height)
    bottom_right_x  = round(p2_frect.pos[0] + half_paddle_width)
    bottom_right_y = round(p2_frect.pos[1] - half_paddle_height)
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 1, -1)
    # draw in ball
    top_left_x = round(b_frect.pos[0]-half_ball_dim)
    top_left_y = round(b_frect.pos[1]+half_ball_dim)
    bottom_right_x = round(b_frect.pos[0] + half_ball_dim)
    bottom_right_y = round(b_frect.pos[1] - half_ball_dim)
    cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 1, -1)
    # downscale # can be downscaled more if necessary...
    return img[::SCALE_FACTOR, ::SCALE_FACTOR].transpose()[:,:,0] # cv2 and what we use use different conventions


def store_frame_info_more_frames(paddle_frect, other_paddle_frect, ball_frect, n_f, table_size):
    global frame
    #global frame_1_info
    global frame_info
    global cur_reward
    global reward_info
    global SCALE_FACTOR
    half_paddle_width = paddle_frect.size[0]/2
    half_paddle_height = paddle_frect.size[1]/2
    half_ball_dim = ball_frect.size[0]/2
    cur_img = render_frame(paddle_frect, other_paddle_frect, ball_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim)
    # plt.imsave('graphics/1.png', img)
    diff_img = cur_img - frame_info[-1] if len(frame_info) > 0 else cur_img # looking at the iamge difference tells us more

    cur_frame_info = []
    if frame == 1:
        for i in range(n_f):
            cur_frame_info.append(diff_img)

    #update reward
    reward_info.append(cur_reward)
    #update_frame_info
    frame_info.append(diff_img)
    return

def check_side(paddle_frect):
    global cur_side
    global last_score

    if paddle_frect.pos[0] < 100:
        side = 'left'
    else:
        side = 'right'

    if side != cur_side:
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

def reset_round():
    global frame
    global cur_reward
    global frame_info
    global reward_info

    frame = 1
    cur_reward = 0
    frame_info = []
    reward_info = []

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def forward_prop(x):
    global params
    x = np.array([x]).T
    h = np.dot(params['W1'], x) + params['b1'] # (H x D) . (D x 1) = (H x 1) (200 x 1)
    h[h<0] = 0 # ReLU introduces non-linearity
    logp = np.dot(params['W2'], h) + params['b2']# This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
    p = sigmoid(logp)  # squashes output to  between 0 & 1 range
    return p

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

    check_side(paddle_frect)
    update_reward(score)
    #store_frame_info(paddle_frect, other_paddle_frect, ball_frect)
    store_frame_info_more_frames(paddle_frect, other_paddle_frect, ball_frect, 75, table_size)
    #decision making by AI
    '''
    if paddle_frect.pos[1]+paddle_frect.size[1]/2 < ball_frect.pos[1]+ball_frect.size[1]/2:
     return "down"
    else:
     return "up"
    '''
    action_prob = forward_prop(frame_info[-1])
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
    params = bt.train_bot(Xtrain, Ytrain, Rtrain, params)
    print("Parameters Updated Successfully!")

def save_params():
    global params
    for key in params:
        params[key] = params[key].tolist()
        #print(type(params[key]))
        #print(params[key])

    filename = 'params.txt'

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

H = TABLE_SIZE[0]//SCALE_FACTOR
D = TABLE_SIZE[1]//SCALE_FACTOR

mode = 'new'
params = {}

if mode == 'new':
    params['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
    params['W2'] = np.random.randn(1,H) / np.sqrt(H) # Shape will be H
    params['b1'] = np.zeros((H,1))
    params['b2'] = np.zeros((1,1))

    init_params = copy.deepcopy(params)
    for key in params:
        init_params[key] = init_params[key].tolist()
    with open('initial_params.txt', 'w') as f:
        f.write(json.dumps(init_params))

elif mode == 'load':

    with open('params_2.txt', 'r') as f:
        params = json.load(f)

    for key in params:
        params[key] = np.array(params[key])











