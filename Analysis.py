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
#          SERVER        #
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

from Agents import FieldPatch, Farmer
from Schedule import RandomActivationByBreed
from SimpleModel import FarmerModel

########################################################
#  TEST THE MODEL FUNCTION                             # 
########################################################
model= FarmerModel(verbose=True,\
                 fake_data = True, 
                 real_data = [None,None],
                 height = 10, \
                 width =10, \
                 initial_farmers =30,\
                 scenario = 'Basic', \
                 index_growth = 0.01)
#Let the model run 20 times 
Data  = model.run_model(step_count=10)


##Print agent transactions 
#agent_transactions  = [farmer.agent_transactions for farmer in \
#                            model.schedule.agents_by_breed[Farmer].values()]
#
#print(agent_transactions)


##########################
# READ REAL DATAFILE     #
##########################
#path_data = 'C:\\Users\\User\\Desktop\\Quyen Nguyen\\Internship\\FarmerModel_Dutchcase\\data\\'
##agent data 
#final_agent_data = pd.read_csv(path_data+ 'final_agent_data.csv')
#print(final_agent_data.head(10))
#
##field data 
#fields_suitability = pd.read_csv(path_data+ 'field_suitability.txt', sep=" ", header=None)
#fields_area = pd.read_csv(path_data+ 'fields_area.txt', sep=" ", header=None)
#fields_ehs = pd.read_csv(path_data+ 'fields_ehs.txt', sep=" ", header=None)
#fields_id = pd.read_csv(path_data+ 'fields_id.txt', sep=" ", header=None)
#fields_le = pd.read_csv(path_data+ 'fields_le.txt', sep=" ", header=None)
#fields_le_current = pd.read_csv(path_data+ 'fields_le_current.txt', sep=" ", header=None)
#fields_le_potential = pd.read_csv(path_data+ 'fields_le_potential.txt', sep=" ", header=None)
#fields_owner = pd.read_csv(path_data+ 'fields_owner.txt', sep=" ", header=None)
#fields_size = pd.read_csv(path_data+ 'fields_size.txt', sep=" ", header=None)
#fields_soil = pd.read_csv(path_data+ 'fields_soil.txt', sep=" ", header=None)
#fields_ehs = pd.read_csv(path_data+ 'fields_ehs.txt', sep=" ", header=None)
#fields_landuse = pd.read_csv(path_data+ 'land_types.txt', sep=" ", header=None)
#
##Combine field data into one single dictionary that include mutliple data tabel
#final_fields_data_dictionary= {'fields_suitability': fields_suitability,\
#                              'fields_area': fields_area,
#                              'fields_ehs': fields_ehs,
#                              'fields_id': fields_id,
#                              'fields_le': fields_le,
#                              'fields_le_current':fields_le_current,
#                              'fields_le_potential':fields_le_potential,
#                              'fields_owner': fields_owner,
#                              'fields_size': fields_size,
#                              'fields_soil': fields_soil,
#                              'fields_ehs': fields_ehs,
#                              'fields_landuse':fields_landuse}

