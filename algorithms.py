import numpy as np
import cost_functions
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import time


def Q_learning(ag, sample, gamma = .5, lr = .5):
    # print(f"sample: {sample}")
    ag.Q[sample['s'],sample['a']] = ((1-lr)*ag.Q[sample['s'],sample['a']]
                                  + lr*(-1*sample['r'] + gamma*ag.Q[sample['sp'],:].max()))


def random_exploration(agents):
    grid_res = 0.1
    step_list = [np.array([grid_res, 0.0]), np.array([-grid_res, 0.0]),
                 np.array([0.0, grid_res]), np.array([0.0, -grid_res])]

    '''
    To cover more area on the board with random steps, we can choose to randomly step
    more than one grid square at once.
    We can still move in bigger steps during exploration, even if we limit our actions 
    during travel to the four adjacent grid squares.

    I take it back - in order to slot the resulting exploration costs into Q(s,a), the 
    action needs to match one of our four adjacent grid squares.

    For now we will assume that the transfer function is deterministic.
    i.e. We only need to learn the cost/reward function R, we don't have to learn T
    '''

    # Initialize Q
    grid_length_x = 20
    grid_length_y = 20
    num_states = (grid_length_x / grid_res) * (grid_length_y / grid_res)
    num_actions = 4
    # Q = np.zeros([len(agents), num_states, num_actions])  # Q[i, :, :] is Q matrix for the ith agent

    k_max = 50000  # number of exploration steps/iterations; we should experiment with this value
    
    for k in range(k_max):
        for idx, ag in enumerate(agents):
            baseline_cost = cost_functions.cost_to_destination(agents)
            current_pos = ag.pos
            state = grid_world_to_state((grid_length_x,
                                         grid_length_y),
                                         *current_pos)

            step_index = np.random.randint(4)  # randomly choose from steps 0,1,2,3
            # step_size = np.random.randint(1,6) # randomly choose number of grid squares to move from 1,2,3,4,5
            step_size = 1  # right now we are only stepping to adjacent grid squares
            random_step = step_size * step_list[step_index]
            ag.move(random_step)
            state_prime = grid_world_to_state((grid_length_x,
                                                grid_length_y),
                                                *ag.pos)

            new_cost = cost_functions.cost_to_destination(agents)
            cost_difference = new_cost - baseline_cost

            sample = {'s': state, 'a': step_index,
                      'r': cost_difference, 'sp': state_prime}
            # action = step_index
            Q_learning(ag, sample)
            # ag.Q[state, action] = cost_difference

            # Update Q somewhere around here, on each exploration step
            # We'll need to convert 2D grid states to linear index of states

    '''
    We need to have a different Q function for each agent, since each of them will require a different policy.
    It is possible to tease out individual rewards using our current cost function, even though it lumps together 
    all the individual costs at once, by only moving one agent at a time and looking at the difference in total cost.

    i.e. Send the current state configuration into the cost function as a baseline. We happen to know that cost only
    depends on agent states, not the actions themselves. Do each action to get a new cost, and subtract this cost
    from the baseline to get the incremental cost associated with that individual agent
    '''

def grid_world_to_state(world_shape, px, py, res=.1):
    update_x = int(px / res)
    update_y = int(py / res)
    update_world_shape = (int(world_shape[0] / res), int(world_shape[1] / res))
    state = np.ravel_multi_index((update_x, update_y), update_world_shape)

    return state

def state_to_grid_world(world_shape, state, res=.1):
    update_world_shape = (int(world_shape[0] / res), int(world_shape[1] / res))
    pos = np.array(np.unravel_index(state, update_world_shape))
    pos[0] *= res
    pos[1] *= res
    return pos

def approximate_Q(agents):
    '''
    We won't be able to reach every action from every state based purely on random exploration.
    Because we are using norms for our cost function, we know that every visited state-action pair in Q will
    have a nonzero, positive value.

    After exploration, we know that any Q(s,a)=0 has not been visited, and we should approximate this entry with
    the existing data that we collected. The simplest method of this approximation is nearest-neighbor (k=1).
    '''
    x = int(agents[0].map_x/agents[0].res)
    y = int(agents[0].map_y/agents[0].res)
    num_states, num_actions = agents[0].Q.shape
    world_shape = (x,y)
    
    start_time = time.time()
    Q_all = agents[0].Q[:,:,np.newaxis]
    for r in range(1,len(agents)):
        Q_all = np.concatenate((Q_all,agents[r].Q[:,:,np.newaxis]),axis=2)
    
    #TODO: code breaks if not all robots have moved in all directions at least once
    #TODO: reason is that code expects at least on action with non-zero reward from each robot to have been taken
    for action in range(0,num_actions):
        states_robots = np.argwhere(Q_all[:,action,:] != 0) # gets all states for all agents for a given action
        states_robots = states_robots[np.argsort(states_robots[:, 1])] # sort them by robot id
        
        explored_coords = np.array(np.unravel_index(states_robots, world_shape,'C')).T[0] # obtain world coords


        max_num = max(np.bincount(states_robots[:,1])) # get maximum number of samples a single robot might have
        
        # calculate indices where padding must be added so vector math can be done
        change_over = np.where(np.roll(states_robots[:,1],1)!=states_robots[:,1])[0]
        change_over = np.repeat(change_over,(max_num-np.bincount(states_robots[:,1])))
        padded_states = np.insert(states_robots[:,0],change_over,0,axis=0).reshape(len(agents),-1)
        padded_coords = np.insert(explored_coords,change_over,[num_states+1,num_states+1],axis=0)


        for state in range(0,num_states):
    
        
            pos = np.array(np.unravel_index(state, world_shape,'C')).reshape(-1,2) # get pos of state currently being examined

            # calculate distances from that state to all sampled states for given action
            distances = np.linalg.norm(pos-padded_coords,axis=1).reshape(len(agents),-1)

            index_array = np.argmin(distances, axis=1) # find index of smallest distance

            # get corresponding state
            closest_states = np.take_along_axis(padded_states, np.expand_dims(index_array, axis=1), axis=1).reshape(-1)
            index = np.arange(0,len(agents),1)
            
            sub_Q_all = Q_all[:,action,:]
            
            loc_of_reward = np.vstack((index,closest_states))
            rewards = sub_Q_all[loc_of_reward[1],loc_of_reward[0]]
            
            empty = np.where(Q_all[state,action,:] == 0, True, False)

            Q_all[state,action,:] = Q_all[state,action,:] + empty*rewards

    for r in range(0,len(agents)):

        agents[r].Q = Q_all[:,:,r]

    print("--- %s seconds ---" % (time.time() - start_time))
    print("for loop finished")



def Q_main(agents):
    print("EXPLORING!")
    random_exploration(agents)
#    print("NEIGHBORS!")
#    approximate_Q(agents)

    '''
    We have two options here: 
    1. We can say our exploration phase is done, and make a fixed policy to follow
       for the rest of the experiment while the agents move toward their destinations.

    2. We can continue to update the Q function after the exploration phase, updating
       on each step we take toward the destination. In this case we need to calculate 
       optimal policy for each state as it comes.

       Would this buy us anything? If each agent is moving in one direction and will
       never traverse the old states behind it again, is it beneficial to update Q
       in it's wake?

    Ideas from Jean: With the knowledge we have of the problem at hand (and also the solution),
            we technically might not gain any benefit from updating Q as we travel. Keep in mind
            though that this only applies if we have a very high confidence that our exploration
            phase was sufficient enough to build up an accurate optimal policy. If we assume that
            our policy is optimal when in reality it isn't, then we may end up re-visiting states
            after all. If we don't update Q, we could think that we're getting certain rewards 
            when we're not where we thought we are.

            That being said, often taking advantage of information about your specific problem can 
            be seen as a "feature". i.e. "We're being smart about our computation time by avoiding
            the need to update Q after the exploration stage"

            In the end, this project is supposed to be enjoyable. You don't want it to be easy peasy
            since there would be no challenge in that, but you also don't need to do a ridiculously
            hard problem for full project credit. Pick the way that makes your life easier. 

    '''
#    print("POLICY!")
#    # Option 1: Find fixed policy from Q
#    for ag in agents:
#        for s in range(ag.num_states):
#            best_action = np.argmax(ag.Q[s, :])
#            ag.Pi[s] = best_action  # this indexes the action in the list of action moves
#    print("EXECUTE!")
#    execute_policy(agents)


def execute_policy(agents):
    # reset agents
    for ag in agents:
        ag.pos = ag.ini_pos.copy()
        ag.traj = np.array([ag.ini_pos.copy()])
    # simulate
    step_list = [np.array([agents[0].res, 0.0]), np.array([-agents[0].res, 0.0]),
                 np.array([0.0, agents[0].res]), np.array([0.0, -agents[0].res])]
    cost = cost_functions.cost_to_destination(agents)
    count = 0
    while cost > 1 and count < 500:
        count += 1
        for ag in agents:
            state = grid_world_to_state((ag.map_x, ag.map_y), *ag.pos)
            step_idx = int(ag.Pi[state])
            ag.move(step_list[step_idx])
        cost = cost_functions.cost_to_destination(agents)

def hooke_jeeves(agents):
    step = 0.1
    step_list = [np.array([step, 0.0]), np.array([-step, 0.0]),
                 np.array([0.0, step]), np.array([0.0, -step])]
    # base cost
    best_cost = cost_functions.cost_to_destination(agents)
    while best_cost > 1:
        for idx, a in enumerate(agents):
            best_step = np.array([0.0, 0.0])
            # step in each direction
            # greedily choose the best move
            for s in step_list:
                a.pos += s
                s_cost = cost_functions.cost_to_destination(agents)
                a.pos -= s
                if s_cost < best_cost:
                    best_cost = s_cost
                    best_step = s
            a.move(best_step, True, True)
            best_cost = cost_functions.cost_to_destination(agents)
