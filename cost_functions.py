import numpy as np
import Agent


def cost_to_destination(agents):
    individual_cost = []
    for agent in agents:
        distance =
        individual_cost.append(np.linalg.norm(agent.pos - agent.des))
        
    overall_cost = np.lnalg.norm(np.nparray(individual_cost))
    
    return overall_cost
        
        
    
        
        
    

