from game_setting import *

game_size = 5 

game = Env(game_size);
control = [0, 1]

Val_func = np.zeros(game_size)
state_counter = np.zeros(game_size)

total_episodes = 10000

policy = np.random.randint(2, size = game_size)

berror = 0.1
discount = 0.5

# Policy iteration

while True:
    print("policy after policy improvement")
    print(policy)
     # Policy evaluation
    while True:
        delta = 0
        for i in range(game_size):
            game.set_pose(i)
            new_state, reward, done = game.move(control[policy[i]])

            prev = Val_func[i]
            if done:
                Val_func[i] = reward
            else:
                Val_func[i] = reward + discount*Val_func[new_state]
        
            diff = abs(prev-Val_func[i])
            
            if diff > delta:
                delta = diff
        if delta < berror:
            break
    
    print("Val_func after policy evalutaion")
    print(Val_func)
   
    # Policy imporvement
    policy_stable = True

    for i in range(game_size):
        temp = np.zeros(len(control))      
        for j in range(len(control)):
            game.set_pose(i)
            new_state, reward, done = game.move(control[j])
            
            if done:
                temp[j] = reward
            else:
                temp[j] = reward + discount*Val_func[new_state]
        
        best_action = rargmax(temp)
        if best_action != policy[i]:
            policy_stable = False
            policy[i] = best_action

    if policy_stable:
        break;

print("Val_func")
print(Val_func)
print("Policy")
print(policy)
 
