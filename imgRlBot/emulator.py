
class pongEmulator:
    def __init__(self, table_size, margins, p1, p2, b):
        self.p1 = fRect(p1.pos, p2.size)
        self.p2 = fRect(p1.pos, p2.size)
        self.b = fRect(b.pos, b.size)
        self.table_size = (table_size[0], table_size[1])
        self.walls_Rects = [Rect((-100, -100), (table_size[0]+200, 100)),
                       Rect((-100, table_size[1]), (table_size[0]+200, 100))]
        self.max_angle=45

    def pump(self):
        pass

    def update(self, p1, p2, b):
        # update pong emulator for when we can get new position data i.e. each time pong_ai is called
        self.p1.pos = p1.pos
        self.p2.pos = p2.pos
        self.b.pos = b.pos
    
    def move_ball(self, velocity): 
        inv_move_factor = int((velocity[0]**2+velocity[1]**2)**.5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                self.step_ball(velocity, 1./inv_move_factor)
        else:
            self.step_ball(velocity, 1)
        
    def step_ball(self, velocity, move_factor):
        #
        moved = 0
        for wall_rect in self.walls_Rects:
            if self.b.get_rect().colliderect(wall_rect):
                c = 0        
                while self.b.get_rect().colliderect(wall_rect):
                    self.b.move_ip(-.1*velocity[0], -.1*velocity[1], move_factor)
                    c += 1 # this basically tells us how far the ball has traveled into the wall
                # r1 = 1+2*(random.random()-.5)*self.dust_error
                # r2 = 1+2*(random.random()-.5)*self.dust_error
                    # r1 = 1 # I think this is correct? Because dust_error is 0
                   #  r2 = 1 
                nv = (velocity[0], -1*velocity[1]) # wallbounce, r1/r2 = 1
                while c > 0 or self.b.get_rect().colliderect(wall_rect):
                    self.b.move_ip(.1*velocity[0], .1*velocity[1], move_factor)
                    c -= 1 # move by roughly the same amount as the ball had traveled into the wall
                moved = 1

        for paddle in [self.p1, self.p2]:
            facing = 1 if paddle.pos[0] > self.table_size[0]/2 else 0
            if self.b.intersect(paddle):
                if (paddle.facing == 1 and self.b.get_center()[0] < paddle.pos[0] + paddle.size[0]/2) or \
                (paddle.facing == 0 and self.b.get_center()[0] > paddle.pos[0] + paddle.size[0]/2):
                    continue
                
                c = 0
                
                while self.b.intersect(paddle) and not self.b.get_rect().colliderect(walls_Rects[0]) and not self.b.get_rect().colliderect(walls_Rects[1]):
                    self.b.move_ip(-.1*velocity[0], -.1*velocity[1], move_factor)
                    
                    c += 1

                #Getting return bounce angle
                # y = distance from centre
                y = self.b.pos[1]+.5*ball.size[1]
                # size = (10, 70) # paddle_size
                #frect.size is same as paddle.size
                center = paddle.pos[1]+paddle.size[1]/2
                rel_dist_from_c = ((y-center)/paddle.size[1])
                rel_dist_from_c = min(0.5, rel_dist_from_c)
                rel_dist_from_c = max(-0.5, rel_dist_from_c)
                sign = 1-2*facing
                theta = sign*rel_dist_from_c*self.max_angle*math.pi/180

                v = velocity

                v = [math.cos(theta)*v[0]-math.sin(theta)*v[1],
                             math.sin(theta)*v[0]+math.cos(theta)*v[1]]

                v[0] = -v[0]

                v = [math.cos(-theta)*v[0]-math.sin(-theta)*v[1],
                              math.cos(-theta)*v[1]+math.sin(-theta)*v[0]]

                # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
                if  v[0]*(2*paddle.facing-1) < 1: # ball is not traveling (a) away from paddle (b) at a sufficient speed
                    v[1] = (v[1]/abs(v[1]))*math.sqrt(v[0]**2 + v[1]**2 - 1) # transform y velocity so as to maintain the speed
                    v[0] = (2*paddle.facing-1) # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to the  increase by 1.2

                #a bit hacky, prevent multiple bounces from accelerating
                #the ball too much
                if not paddle is self.prev_bounce:
                    velocity = (v[0]*self.paddle_bounce, v[1]*self.paddle_bounce)
                else:
                    velocity = (v[0], v[1])
                self.prev_bounce = paddle
                

                while c > 0 or self.frect.intersect(paddle.frect):
                
                    self.frect.move_ip(.1*velocity[0], .1*velocity[1], move_factor)
                    
                    c -= 1
                
                moved = 1
                

        if not moved:
            self.frect.move_ip(velocity[0], velocity[1], move_factor)



class fRect:
    """Like PyGame's Rect class, but with floating point coordinates"""
    def __init__(self, pos, size):
        self.pos = (pos[0], pos[1])
        self.size = (size[0], size[1])
    def move(self, x, y):
        return fRect((self.pos[0]+x, self.pos[1]+y), self.size)

    def move_ip(self, x, y, move_factor = 1):
        self.pos = (self.pos[0] + x*move_factor, self.pos[1] + y*move_factor)

    def get_rect(self):
        return Rect(self.pos, self.size)

    def copy(self):
        return fRect(self.pos, self.size)

    def intersect(self, other_frect):
        # two rectangles intersect iff both x and y projections intersect
        for i in range(2):
            if self.pos[i] < other_frect.pos[i]: # projection of self begins to the left
                if other_frect.pos[i] >= self.pos[i] + self.size[i]:
                    return 0
            elif self.pos[i] > other_frect.pos[i]:
                if self.pos[i] >= other_frect.pos[i] + other_frect.size[i]:
                    return 0
        return 1 #self.size > 0 and other_frect.size > 0

    def get_center(self):
        return (self.pos[0] + .5*self.size[0], self.pos[1] + .5*self.size[1])








