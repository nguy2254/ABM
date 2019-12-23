"""
REPLICATION OF THE ABM MODEL 
An empirical ABM for regional land use/cover change: 
a Dutch case study (version 1.0.0)
Quyen Nguyen | 16 December 2019 
THE MODEL INCLUDE FIVE COMPONENTS
1. Agents: The module defines the characteristics and actions of the environment 
(Field Patch) and the farmers (Farmer)
2. Schedule: The moduel defines the order that the model runs in cases there are 
several agent breeds. There are 2 breeds in the base model (Farmer and Field Patch).
The Farmer can make decision, the Field Patch is static and only change after their 
respective owning Farmer makes decision
3. SimpleModel: The module defines the inital states and the steps that all farmers 
makes during the time t 
4. Server: The module defines the visualization of the model
5. Analysis: The module shows the analysis of the model 
##########################
#          SCHEDULE      #
##########################  
"""
##########################
# IMPORT GENERIC LIBRARY #
##########################
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation 
from mesa.space import MultiGrid
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector
from collections import defaultdict 
from mesa.visualization.ModularVisualization import ModularServer 
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter
import random
import sys 
import os
from random import gauss
import datetime

##########################
# IMPORT MODEL COMPONENT #
##########################
import sys 
import os
path = 'C:\\Users\\User\\Desktop\\Quyen Nguyen\\Internship\\FarmerModel_Dutchcase\\'

sys.path.append(os.path.abspath(path))

#Create Schedule 
class RandomActivationByBreed(RandomActivation):
    #it is a scheduler that activate each type of agent once per step 
    def __init__(self,model): 
        super().__init__(model)
        self.agents_by_breed = defaultdict(dict)

    #add agent 
    def add(self,agent): 
        self._agents[agent.unique_id] = agent 
        agent_class = type(agent)
        self.agents_by_breed[agent_class][agent.unique_id] = agent 

    #remove agent 
    def remove(self,agent): 
        del self._agents[agent.unique_id]
        agent_class = type(agent)
        del self.agents_by_breed[agent_class][agent.unique_id]

    #execute the step of each agent breed one time in random order 
    def step(self,by_breed=True):
        if by_breed: 
            for agent_class in self.agents_by_breed: 
                self.step_breed(agent_class)
            self.steps += 1
            self.time += 1 
        else: 
            super().step()
    
    #get a list of objects by agent type 
    def get_agents_by_breed(self, breed_class):
        return self.agents_by_breed[breed_class]
    
    #shueffle order and run all agents of a given breed 
    def step_breed(self, breed): 
        agent_keys = list(self.agents_by_breed[breed].keys())
        self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            self.agents_by_breed[breed][agent_key].step()
            
    #count the number of agents within each breed 
    def get_breed_count(self, breed_class):
        return len(self.agents_by_breed[breed_class].values())

