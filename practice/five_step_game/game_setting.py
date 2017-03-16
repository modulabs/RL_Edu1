import tensorflow as tf
import numpy as np
import sys,tty,termios

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
        print("Game start") 
        self.steps = steps
        self.pose = int(steps/2)
        self.print_state()
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
            reward 
            1 : game terminate win
            -1 : game terminate lose
            0 : continue game 
        '''
        if control==1: 
            self.pose += 1
            if self.pose == self.steps:
                return 1
            return 0
        elif control==0:
            self.pose -=1
            if self.pose == -1:
                return -1
            return 0
        else :
            print("Wrong control : {}".format(control))
            return 0

game = Env(5);
while True:
    control = get()
    if control==-1:
        break
    reward = game.move(control)
    game.print_state()
    print("Reward : {}".format(reward))
    if reward == 1:
        print("Game win")
        break;
    if reward == -1:
        print("Game lose")
        break;
   

