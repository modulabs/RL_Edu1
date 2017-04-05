import tensorflow as tf
import numpy as np

class Env(object):
    def __init__(self, steps = 5):
        print("Game start") 
        self.steps = steps
        self.pose = int(steps/2)
        self.print_state()
    def reset(self):
        self.pose = int(self.steps/2)
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
control = [0, 1]

Val_func = np.zeros(game.steps)
state_counter = np.zeros(game.steps)

total_episodes = 10000

for i in range(total_episodes):
    game.reset()    
    state_track = []
    state_track.append(game.pose)
    done = False
    while True:
        control_index = np.random.randint(len(control))
        new_state, reward, done = game.move(control[control_index])
        if done == True:
            for j in range(len(state_track)):
                state_counter[state_track[j]]+= 1
                Val_func[state_track[j]]+= reward
            break
        else:
            state_track.append(new_state)

for i in range(game.steps):
    if state_counter[i] is not 0:
        Val_func[i] /= state_counter[i]


print(Val_func)
