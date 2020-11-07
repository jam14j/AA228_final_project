import Agent
import cost_functions
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
        print(a.ini_pos)
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

    for idx in range(agent_count):
        pos = np.array([np.cos(idx*np.pi/6), np.sin(idx*np.pi/6)])
        des_pos = np.array([10*np.cos((idx+6)*np.pi/6), 10*np.sin((idx+6)*np.pi/6)])
        # spiral
        # pos = np.array([0.0, 0.0])
        # des_pos = np.array([idx*np.cos((idx+6)*np.pi/6), idx*np.sin((idx+6)*np.pi/6)])

        #des_pos = np.array([x[idx], y[idx]])
        agents[idx] = Agent.Agent(pos, des_pos)

    alg.hooke_jeeves(agents)
    # Q_learning(agents)
    plot_agents(agents)

if __name__ == "__main__":
    main()
