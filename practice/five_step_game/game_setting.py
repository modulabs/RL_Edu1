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
            new_state:
                new_position 
            reward :
                1 : game terminate win
                0 : otherwise 
            done : 
                True : done 
                False : Not done
        '''
        if control==1: 
            self.pose += 1
            if self.pose == self.steps:
                done = True 
                reward = 1
                new_state = self.pose
                return new_state, reward, done
            
            done = False
            reward = 0
            new_state = self.pose
            return new_state, reward, done
        elif control==0:
            self.pose -=1
            if self.pose == -1:
                done = True
                reward = 0
                new_state = self.pose
                return new_state, reward, done
            done = False
            reward = 0
            new_state = self.pose
            return new_state, reward, done
           
        else :
            print("Wrong control : {}".format(control))
            done = False
            reward = 0
            new_state = self.pose
            return new_state, reward, done

game = Env(5);
done = False
while not done:
    control = get()
    if control==-1:
        break
    new_state, reward,done = game.move(control)
    game.print_state()
    print("new_state, reward, done : {}, {}, {}".format(new_state, reward))
