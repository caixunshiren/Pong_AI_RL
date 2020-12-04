import math
import Pong_Game
import RLbot

'''
a = Pong_Game.fRect((200, 200), (15,15))
b = Pong_Game.fRect((202, 200), (15,15))
c = Pong_Game.fRect((204, 200), (15,15))
ball_pos_history = [a, b, c]
'''
ball_pos_history = [(200,200), (202,200), (204,200)] # [(x, y), (x,y) ..



predicted_pos = 133+7

#cache = {"v":[], "n":0, "d1":0, "a":0, "p1":0, "p2":0, "table_size":0, "case":0 }

state = "new_game"

def reinit():
    global ball_pos_history
    ball_pos_history = [(200,200), (202,200), (204,200)]

def get_velocity(p1, p2):
    return ((p2[0]-p1[0], p2[1]-p1[1]))


def if_flip(ph):
    return (get_velocity(ph[-2], ph[-3])[0] < 0 and get_velocity(ph[-3], ph[-4])[0] > 0) or (get_velocity(ph[-2], ph[-3])[0] > 0 and get_velocity(ph[-3], ph[-4])[0] < 0)

def predict_position(p1, p2, table_size, h):
    # returns distance between y = 0 and predicted final position when it "scores"
    #   /<-
    #  /
    # /
    #. p1
    # \
    #  \
    #   \-> p2
    #print("predictied height:", h)
    v = get_velocity(p1, p2)

    table_size = (table_size[0]-70, table_size[1]-15)
    try:
        v = list(v)
        v[0] = abs(v[0])
        #if no bounce
        '''
        if abs((p1[1])*(v[0]/v[1])) > table_size[0]:
            cache["v"] = v
            cache["n"] = "na"
            cache["d1"] = "na"
            cache["a"] = "na"
            cache["p1"] = p1
            cache["p2"] = p2
            cache["table_size"] = table_size
            return p1[1]+7.5+table_size[0]*(v[1]/v[0])
        '''
        if v[1] < 0 and abs(table_size[0]*(v[1]/v[0])) < p1[1]:
            #cache["v"] = v
            #cache["n"] = "na"
            #cache["d1"] = "na"
            #cache["a"] = "na"
            #cache["p1"] = p1
            #cache["p2"] = p2
            #cache["table_size"] = table_size
            return p1[1]+7.5+table_size[0]*(v[1]/v[0])

        if v[1] > 0 and abs(table_size[0]*(v[1]/v[0])) < table_size[1] - p1[1]:
            #cache["v"] = v
            #cache["n"] = "na"
            #cache["d1"] = "na"
            #cache["a"] = "na"
            #cache["p1"] = p1
            #cache["p2"] = p2
            #cache["table_size"] = table_size
            return p1[1]+7.5+table_size[0]*(v[1]/v[0])

        d1 = v[0]/abs(v[1])
        #number of bounces
        if v[1] > 0:
            d1 = (table_size[1] - p1[1])*d1
        else:
            d1 = p1[1]*d1
        #print("d1 is :", d1)

        n = (table_size[0] - d1) // (table_size[1]*(v[0]/abs(v[1]))) + 1
        #print("n is :", n)

        a = (table_size[0] - d1) % (table_size[1]*(v[0]/abs(v[1])))
        #print("a is :", a)

        #store to cache
        #cache = {"v":[], "n":0, "d1":0, "a":0, "p1":0, "p2":0, "table_size":0 }
        #cache["v"] = v
        #cache["n"] = n
        #cache["d1"] = d1
        #cache["a"] = a
        #cache["p1"] = p1
        #cache["p2"] = p2
        #cache["table_size"] = table_size

        #cases
        if n%2 == 0 and v[1] > 0:
            #k = 7.5 + a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 1:", k)
            #cache["case"] = "case 1"
            return 7.5 + a*(abs(v[1])/v[0])

        if n%2 == 0 and v[1] < 0:
            #k = 272.5 - a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 2:", k)
            #cache["case"] = "case 2"
            return 272.5 - a*(abs(v[1])/v[0])

        if n%2 == 1 and v[1] > 0:
            #k = 272.5 - a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 3:", k)
            #cache["case"] = "case 3"
            return 272.5 - a*(abs(v[1])/v[0])

        if n%2 == 1 and v[1] < 0:
            #k = 7.5 + a*(abs(v[1])/v[0])
            #if k < 0 or k > 280: print("case 4:", k)
            #cache["case"] = "case 4"
            return 7.5 + a*(abs(v[1])/v[0])

    except:
        print("exception")
        return predicted_pos

def get_sqr_dist(a, b):
    return (a.pos[0]-b.pos[0])**2 + (a.pos[1]-b.pos[1])**2

def check_state(paddle_frect, other_paddle_frect, ball_frect):
    global state
    #print("flip")
    if get_sqr_dist(paddle_frect, ball_frect) < get_sqr_dist(other_paddle_frect, ball_frect):
        state = "chaser_mode"
        #print("switched to:",state)

    else:
        state = "predict_mode"
        #print("switched to:",state)

def check_win(paddle_frect, other_paddle_frect, ball_frect):
    global ball_pos_history
    global state
    if paddle_frect.pos[0] > ball_frect.pos[0] and paddle_frect.pos[0] < 100:
        #print("opponent win")
        ball_pos_history = [(200,200), (202,200), (204,200)]
        state = "new_game"
        #print_cache()
    elif paddle_frect.pos[0] < ball_frect.pos[0] and paddle_frect.pos[0] > 300:
        #print("opponent win")
        ball_pos_history = [(200,200), (202,200), (204,200)]
        state = "new_game"
        #print_cache()
    elif other_paddle_frect.pos[0] > ball_frect.pos[0] and other_paddle_frect.pos[0] < 100:
        #print("you win")
        ball_pos_history = [(200,200), (202,200), (204,200)]
        state = "new_game"
    elif other_paddle_frect.pos[0] < ball_frect.pos[0] and other_paddle_frect.pos[0] > 300:
        #print("you win")
        ball_pos_history = [(200,200), (202,200), (204,200)]
        state = "new_game"

def print_cache():
    global predicted_pos
    print(cache)
    print(predicted_pos)

def pongbot(paddle_frect, other_paddle_frect, ball_frect, table_size, score):
    global ball_pos_history # wish we had classes
    global predicted_pos
    global state
    '''return "up" or "down", depending on which way the paddle should go to
    align its centre with the centre of the ball, assuming the ball will
    not be moving

    Arguments:
    paddle_frect: a rectangle representing the coordinates of the paddle
                  paddle_frect.pos[0], paddle_frect.pos[1] is the top-left
                  corner of the rectangle.
                  paddle_frect.size[0], paddle_frect.size[1] are the dimensions
                  of the paddle along the x and y axis, respectively

    other_paddle_frect:
                  a rectangle representing the opponent paddle. It is formatted
                  in the same way as paddle_frect
    ball_frect:   a rectangle representing the ball. It is formatted in the
                  same way as paddle_frect
    table_size:   table_size[0], table_size[1] are the dimensions of the table,
                  along the x and the y axis respectively

    The coordinates look as follows:

     0             x
     |------------->
     |
     |
     |
 y   v
    '''

    RLbot.pongbot(paddle_frect, other_paddle_frect, ball_frect, table_size, score)

    ball_pos_history.append(ball_frect.pos)

    #if state == "new_game" and ball_pos_history[-3][0] > 100 and ball_pos_history[-3][0] < 300:
    #    predicted_pos = predict_position(ball_pos_history[-2], ball_pos_history[-1], table_size, ball_pos_history[-3][1])

    if if_flip(ball_pos_history): # could make slightly faster by calculating get_velocity in outside loop and passing v to func instead
        #print("flip!")

        check_state(paddle_frect, other_paddle_frect, ball_frect)

        v = get_velocity(ball_pos_history[-2], ball_pos_history[-1]) # -'ve x velocity means going to the left
        # update predicted_pos only when opponent hits the ball
        if (((v[0] < 0) and (paddle_frect.pos[0] < table_size[0]/2)) or ((v[0]>0) and (paddle_frect.pos[0]>table_size[0]/2))):
            predicted_pos = predict_position(ball_pos_history[-2], ball_pos_history[-1], table_size, ball_pos_history[-3][1])

    check_win(paddle_frect, other_paddle_frect, ball_frect)

    if state == "chaser_mode" or state == "new_game":
        # do one round of training
        RLbot.update(score)
        return pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size)

    return controller(predicted_pos, paddle_frect.pos[1], paddle_frect, other_paddle_frect, ball_frect, table_size, score)

def controller(desired_pos, current_pos, paddle_frect, other_paddle_frect, ball_frect, table_size, score):

    if 35 > abs(desired_pos-35 - current_pos):
        ret = RLbot.get_ret()
        RLbot.update(score)
        return ret
    else:
        RLbot.update(score)
        if current_pos < desired_pos-35:
            return "down"
        else:
            return "up"

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):

    if paddle_frect.pos[1]+paddle_frect.size[1]/2 < ball_frect.pos[1]+ball_frect.size[1]/2:
     return "down"
    else:
     return "up"
