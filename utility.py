'''
Utility functions that are useful for other functions
'''

# Imports
import random

from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.env.agent import Agent

def get_existing_models(modellist_path=None):
    '''
    Reads the modellist file given and returns the list of models in the file
    If file does not exist, returns an empty list.
    '''
    if not modellist_path:
        modellist_path = "C:/Users/lenovo/Desktop/Coding/LuxAI/rl/modelcp/modellist.txt"
    try:
        with open(modellist_path, 'r') as f:
            modellist =  f.read().split('\n')
        modellist.remove('')
    except FileNotFoundError:
        print(f"File not found. Path: {modellist_path}")
        with open(modellist_path, 'w') as f:
            f.write('')
        modellist = []
    return modellist

def is_same(model1, model2, num_envs=3) -> bool:
    '''
    Function to test if 2 models are the same. Generates 1 environment and use both models to predict on the same env.
    Then, compare the predictions made.
    '''
    from models.agent_policy import AgentPolicy

    for _ in range(num_envs):
        configs = LuxMatchConfigs_Default
        configs['seed'] = random.randint(0, 10000)
        env = LuxEnvironment(configs, AgentPolicy(), Agent())
        obs = env.reset()
        for i in range(600):
            action_code1, _status1 = model1.predict(obs, deterministic=True)
            action_code2, _status2 = model2.predict(obs, deterministic=True)
            if action_code1 != action_code2:
                print(obs, action_code1, action_code2)
                return False
            obs, reward, done, info = env.step(action_code1)
            if done:
                break
    return True