# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 20:39:01 2018

@author: Zymieth
"""

import numpy as np
import matplotlib

def set_true_value_of_actions(size):
    """Takes as input the size of the action space and returns a randomly generated list
        of the true values"""
    q_star = []
    
    [q_star.append(np.random.uniform(-10,10)) for i in range(size)]
    return q_star

def action_values(size):
    return np.zeros(size)

def action_counts(size):
    return np.zeros(size)

def reward_function(q_star, a):
    """Given an action 'a' returns its corresponding reward samples around q_star(a)"""
    reward_distributions = []
    
    [reward_distributions.append(np.random.normal(loc=q_star[i])) for i in range(len(q_star))]
    return reward_distributions[a]

epsilon = 0.01
elements = [1, 0]
probabilities = [1 - epsilon, epsilon]


#k-bandits algorithm
n_actions = 10

a_values = action_values(n_actions)
counts = action_counts(n_actions)
q_star = set_true_value_of_actions(n_actions) 

for i in range(1000):
    e_greedy_step = np.random.choice(elements, 1, p=probabilities) 
    if e_greedy_step == 1:
        action = int(np.max(a_values))
        counts[action] += 1
        a_values[action] =+ (reward_function(q_star, action) -
                a_values[action])/counts[action]
        
    else:
        action = np.random.randint(0,n_actions-1)
        counts[action] += 1
        a_values[action] =+ (reward_function(q_star, action) -
                a_values[action])/counts[action]
    
    print(a_values)
    matplotlib.pyplot.hist(a_values)
        