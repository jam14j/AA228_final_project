import Agent
import numpy as np
import matplotlib.pyplot as plt
import algorithms as alg
from snowflake import koch_snowflake

# Set random seed for reproducible results
np.random.seed(1)

def print_agents(agents):
    print("idx\tpos\tdes_pos")
    for idx, a in enumerate(agents):
        print(f"{idx}\t{a.pos}\t{a.des_pos}")

def plot_agents(agents):
    for a in agents:
        plt.plot(a.ini_pos[0], a.ini_pos[1], 'bo')
        plt.plot(a.traj[:,0], a.traj[:,1])
        plt.plot(a.des_pos[0], a.des_pos[1], 'r*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


def main():
    # snowflake
    #x, y = koch_snowflake(order=2)
    #agent_count = len(x)
    #plt.plot(x,y)

    agent_count = 12
    agents = [0.0 for _ in range(agent_count)]

    pos_array = np.zeros((len(agents), 2))
    des_pos_array = np.zeros((len(agents), 2))
    for idx in range(agent_count):
        pos = np.array([np.cos(idx*np.pi/6), np.sin(idx*np.pi/6)])
        des_pos = np.array([10*np.cos((idx+6)*np.pi/6), 10*np.sin((idx+6)*np.pi/6)])
        # spiral
        # pos = np.array([0.0, 0.0])
        # des_pos = np.array([idx*np.cos((idx+6)*np.pi/6), idx*np.sin((idx+6)*np.pi/6)])

        #des_pos = np.array([x[idx], y[idx]])
        agents[idx] = Agent.Agent(pos, des_pos)
        pos_array[idx, :] = agents[idx].pos
        des_pos_array[idx, :] = agents[idx].des_pos

    min_pos = np.min(pos_array)
    min_des_pos = np.min(des_pos_array)
    # print(f"min_des: {min_pos}\nmin_des_pos: {min_des_pos}")
    min_value = np.abs(np.min((min_pos,min_des_pos)))

    for a in agents:
        a.traj[0,:] += np.array([min_value, min_value])
        a.ini_pos += np.array([min_value, min_value])
        a.pos += np.array([min_value, min_value])
        a.des_pos += np.array([min_value, min_value])

    # alg.hooke_jeeves(agents)
    alg.Q_main(agents)
    plot_agents(agents)

if __name__ == "__main__":
    main()
