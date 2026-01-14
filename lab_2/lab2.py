import os, sys, random
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from utils.ai_lab_functions import UCT_log
import gym, envs
from timeit import default_timer as timer
import numpy as np
import random
import math

env = gym.make("ProbabilisticPlanningDomain-v0")

print("Number of actions: ", env.action_space.n)
print("Actions: ", env.actions)
print("States: ", env.states)
print("Probability from d1 to d2 with action m12:", env.Pr[('d1', 'm12', 'd2')])
print("Actions that can be performed in state d2:", env.Applicable('d2'))
print("States in which action m23 performed in state d2 could end:", env.gamma('d2', 'm23'))
print("Cost (already converted into a negative reward) for performing m12 from d1 to d2:", env.get_cost('d1', 'm12', 'd2'))

def policy_iteration(environment, maxiters=150, maxviters=20, delta=1e-3):
    """
    Performs the policy iteration algorithm for a specific environment
    
    Args:
        environment: environment
        maxiters: timeout for the iterations
        maxviters: timeout for policy evaluation iterations
        delta: policy evaluation convergence parameter
        
    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """
    i = 1
    pi = environment.init_safe_policy()
    V = dict()        
    
    while i <= maxiters:
        print("iteration", i)
        i += 1
        j = 1
        for state in environment.states:
            V[state] = 0
        
        # 1) Policy evaluation
        while j <= maxviters:

            #
            #  YOUR CODE HERE
            #

        #2) Policy Improvement
        unchanged = True  
        for state in environment.states:
            if state == env.goal_state:
                continue
            
            #
            #  YOUR CODE HERE
            #
            
        if unchanged or maxiters == 0: 
            print(f"STOP [unchanged = {unchanged}]")
            break       
    
    return np.asarray(pi)

env = gym.make("ProbabilisticPlanningDomain-v0")
env.reset()

t = timer()
print("\nINITIAL SAFE POLICY: \n{}".format(env.init_safe_policy()))

policy = policy_iteration(env)

print("\nEXECUTION TIME: \n{}".format(round(timer() - t, 4)))
print(policy)

def Sample(sim_env, s, a):
    sim_env.state = s
    next_state, _, _, _ = sim_env.step(a)
    return next_state
def V0(s):
    return 0
def Select(sim_env, s, Q, N):
    c = math.sqrt(2)
    return max(sim_env.Applicable(s), key=lambda a: Q[s].get(a, 0) - c * math.sqrt(math.log(N[s]) / (1 + N.get((s, a), 0))))


def UCT_rollout(s, h, sim_env, Envelope=None, Q=None, N=None, metrics=None):
    """
    UCT-Rollout implementation.
    
    Parameters:
        s: Current state
        h: Current depth
        sim_env: Environment 
        Envelope: Set to store visited states
        Q: Dictionary for Q-values
        N: Dictionary for visit counts
        metrics: Dictionary to track evaluation metrics
    """
    # Initialize Envelope, Q, and N if not provided
    if Envelope is None:
        Envelope = set()
    if Q is None:
        Q = {}
    if N is None:
        N = {}

    # Base cases
    if s in sim_env.S_g():
        return 0
    if h == 0:
        return V0(s)
    
    if s not in Envelope:
        Envelope.add(s)
        N[s] = 0
        Q[s] = {}
        for a in sim_env.Applicable(s):
            Q[s][a] = 0
            N[(s,a)] = 0

    #
    #  YOUR CODE HERE
    #

    return cost_rollout


import copy
import time
def UCT_Lookahead(s, h, n_sim, env, metrics):
    """
    UCT algorithm implementation.

    Parameters:
        s: Initial state
        h: Depth limit
        n_sim: The number of simulations to perform, this represents the termination condition
        env: environment in which to perform the procedure
        metrics: dictionary of evaluation metrics
    """
    if len(env.Applicable(s)) != 0:
        sim_env = copy.deepcopy(env)
        Q = {}
        N = {}
        Envelope = set()
        start_time = time.time()
        for _ in range(n_sim):
            metrics["total_iterations"] += 1
            UCT_rollout(s, h, sim_env, Envelope, Q, N, metrics)
        metrics["total_time"] += time.time() - start_time
        if s in Q:
            best_action = max(Q[s], key=Q[s].get)
            return best_action
    else:
        print("Dead end state reached")
        return None
    

def MDP_Lookahead(env, depth, simulations):

    eval_metrics = {
        "total_iterations": 0,
        "total_time": 0,
        "new_action_count": 0,
        "tried_action_count": 0
    }
    
    state = env.reset()          
    done = False
    while not done:
        action = UCT_Lookahead(state, depth, simulations, env, eval_metrics)
        print(UCT_log(env=env, state=state, action=action))
        state, _, done, _ = env.step(action)
        if done:
            print("Episode terminated in state ", state)

    return state, eval_metrics


env = gym.make("ProbabilisticPlanningDomain-v0")
state, metrics = MDP_Lookahead(env, 50, 50)
if state == env.goal_state:    
    print("\nGoal state reached!")

print("\n--- Evaluation Metrics ---")    
print(f"Total Iterations: {metrics['total_iterations']}")
print(f"Total Time (seconds): {metrics['total_time']:.4f}")
print(f"New Action Count: {metrics['new_action_count']}")
print(f"Tried Action Count: {metrics['tried_action_count']}")