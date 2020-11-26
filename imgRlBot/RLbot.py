import numpy as np
import copy
import cv2
import conv_trainer as ct
from utils import render_frame
from collections import deque
frame = 1
cur_side = 'left'
cur_reward = 0
last_score = [0,0]
reset = False

train = []
Ytrain = []
Rtrain = []
SCALE_FACTOR = 5
TABLE_SIZE = (440, 280)

class gameData:
    def __init__(self):
        self.img_list = deque()
        self.xtrain = deque()
        self.ytrain = deque()
        self.cur_side = 'left'
        self.last_score = [0,0]
        self.cur_reward = 0
        self.frame = 1
        self.reset=False
    def export_numpy(self):
        return np.array(self.img_list), np.array(self.xtrain), np.array(self.ytrain)

class mdlmng:


gd = gameData()






def update_reward(score):
    global gd
    if score[0] == gd.last_score[0] and score[1] == gd.last_score[1]:
        gd.reward = 0
    else:
        gd.reset = True
        if gd.cur_side == 'left':
            gd.cur_reward = score[0] - gd.last_score[0] + gd.last_score[1] - score[1]
        else:
            gd.cur_reward = score[1] - gd.last_score[1] + gd.last_score[0] - score[0]


def train_pongbot(paddle_frect, other_paddle_frect, ball_frect, table_size, score = []):
    global gd
    gd.cur_reward = check_side(paddle_frect)
    update_reward(score)
    half_paddle_width = paddle_frect.size[0]/2
    half_paddle_height = paddle_frect.size[1]/2
    half_ball_dim = ball_frect.size[0]/2
    cur_img = render_frame(paddle_frect, other_paddle_frect, ball_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim)
    diff_img = cur_img - gd.img_list[-1] if len(gd.img_list) > 0 else cur_img # looking at the iamge difference tells us more
    gd.img_list.append(diff_img)


    action_prob = forward_prop(frame_info[-1]) # 1 x 88 ?? 

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
    x = np.array(x).T 
    h = np.dot(params['W1'], x) + params['b1'] # (H x D) . (D x H) (88 x 56) . (56 x 88) = (88 x 88) 
    h[h<0] = 0 # ReLU introduces non-linearity # (88 x 88)
    h2 = np.dot(params['W2'], h) + params['b2'] # (1 x 88) . (88 x 88 = (1 x 88)
    logp = np.dot(params['W3'], h2.T) + params['b3'] # (1 x 88) x (88 x 1) = 1 (scalar)
    p = sigmoid(logp)  # squashes output to  between 0 & 1 range
    return p



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










