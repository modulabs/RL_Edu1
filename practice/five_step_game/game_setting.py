import tensorflow as tf
import numpy as np
import sys,tty,termios

def rargmax(vector):
    '''
    input :
        1D vector 
    return :
        one of the max indices
    '''
    m = np.max(vector)
    indices = np.transpose(np.nonzero(vector == m)[0])#Get every max indices
    return indices[np.random.randint(len(indices))]#return one of the value


class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def get():
    inkey = _Getch()
    while(1):
        k=inkey()
        if k!='':break
    if k=='\x1b[C':
        print("right")
        return 1
    elif k=='\x1b[D':
        print("left")
        return 0
    else:
        print("not an arrow key!")
        return -1

class Env(object):
    def __init__(self, steps = 5):
        # pose : [0, 1, ..., steps-1] 
            #pose -1 : L
            #pose steps : W
        # reward 2D array pose*control
        
        print("Game start") 
        self.steps = steps
        self.pose = int(steps/2)
        
        self.r = np.zeros((self.steps, 2))
        self.r[self.steps-1][1] = 1
        
        self.print_state()

    def set_pose(self, pose):
        if pose >= self.steps or pose<=-1:
            print("wrong pose")
            return
        self.pose = pose

    def reward(self, pose, control):
        return self.r[pose][control]

    def reset(self):
        self.set_pose(int(self.steps/2))

    def print_state(self):
        '''
        L001...00W
        '''
        positions = np.zeros(self.steps)
        if self.pose >= 0 and self.pose < self.steps:
            positions[self.pose] = 1
        s = ""
        for i in range(self.steps):
            if positions[i]==0:
                s = s+"0"
            else:
                s = s+"1"
        print("L"+s+"W")
    
    def move(self, control):
        '''
        input :
            control
        1 : move right
        0 : move left
        return :
            new_state:
                new_position 
            reward :
                1 : game terminate win
                0 : otherwise 
            done : 
                True : done 
                False : Not done
        '''
        if control==1 or control ==0:
            reward = self.reward(self.pose, control)
            self.pose += 2*control-1
            new_state = self.pose
            done = False
            if self.pose == self.steps or self.pose==-1:
                done = True 
            return new_state, reward, done
        else :
            print("Wrong control : {}".format(control))
            done = False
            reward = 0
            new_state = self.pose
            return new_state, reward, done


if __name__ == "__main__":
    game = Env(5);
    done = False
    while not done:
        control = get()
        if control==-1:
            break
        new_state, reward,done = game.move(control)
        game.print_state()
        print("new_state, reward, done : {}, {}, {}".format(new_state, reward, done))
