from game_setting import *

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
