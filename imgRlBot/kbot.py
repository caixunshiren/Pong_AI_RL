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
        self.rtrain = deque()

        self.round_x = deque()
        self.round_y = deque()
        self.round_reward = deque()

        self.cur_side = 'left'
        self.last_score = [0,0]
        self.cur_reward = 0
        self.frame = 1
        self.reset=False
    
    def export_numpy(self):
        return np.array(self.img_list), np.array(self.xtrain), np.array(self.ytrain)

class mdlmngr:
    def __init__(self, run_model, train_model):
        self.run_model = run_model
        self.train_model = train_model
    def save_model(self):
        

#############
# Main block
gd = gameData()

# load models...

mm = mdlmngr(rm, tm)
#############

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
    global gd, SCALE_FACTOR, mm
    gd.cur_reward = check_side(paddle_frect)
    update_reward(score)
    half_paddle_width = paddle_frect.size[0]/2
    half_paddle_height = paddle_frect.size[1]/2
    half_ball_dim = ball_frect.size[0]/2
    cur_img = render_frame(paddle_frect, other_paddle_frect, ball_frect, table_size, half_paddle_width, half_paddle_height, half_ball_dim, SCALE_FACTOR, table_size)
    diff_img = cur_img - gd.img_list[-1] if len(gd.img_list) > 0 else cur_img # looking at the iamge difference tells us more
    gd.img_list.append(diff_img)

    action_prob = mm.run_model.predict(np.expand_dims(diff_img, axis=0), batch_size)[0][0]
    action = np.random.choice(a=[2,3],size=1,p=[action_prob, 1-action_prob])
    ret = 'up' if action == 2 else 'down'
    y = 1 if ret == 'up' else 0
    
    gd.ytrain.append(y)
    # frame info is a list of each round 

    gd.frame += 1
    gd.last_score = copy.deepcopy(score)
    if reset:
        #pass info to training
        gd.xtrain.append(gd.round_x)
        gd.Rtrain.append(gd.round_r)
        gd.ytrain.append(gd.round_y)
        reset_round()
        reset = False
    
    return ret

def check_side(paddle_frect):
    global gd
    if paddle_frect.pos[0] < 100:
        side = 'left'
    else:
        side = 'right'
    if side != gd.cur_side:
        gd.cur_side = side
        gd.last_score = [0,0]

def reset_round():
    global gd
    gd.frame = 1
    gd.cur_reward = 0
    gd.round_x.clear()
    gd.round_y.clear()
    gd.round_r.clear()


