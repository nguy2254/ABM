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
#  DEFINE VISUALIZATION COMPONENT                      #
########################################################
def FarmerPotrayal(agent):
    if agent is None:
        return
    portrayal = {}
    if type(agent) is Farmer:
        portrayal["Filled"]= "true"
        portrayal["r"]= 0.5
        portrayal["scale"] = 1
        portrayal["Layer"]= 1
        portrayal["Shape"]= 'circle'
        if agent.agent_type ==1:
            portrayal["Color"]=  "red"
#            portrayal["text"] = 'hobby'
#            portrayal["text_color"] = None
        elif agent.agent_type ==2:
            portrayal["Color"]=  "blue"
#            portrayal["text"] = 'conventional'
#            portrayal["text_color"] = None
        elif agent.agent_type ==3:
            portrayal["Color"]=  "black"
#            portrayal["text"] = 'diversifier'
#            portrayal["text_color"] = None
        elif agent.agent_type ==4:
            portrayal["Color"]=  "orange"
#            portrayal["text"] = 'conventional expansionist'
#            portrayal["text_color"] = None
        elif agent.agent_type ==5:
            portrayal["Color"]=  "gray"
#            portrayal["text"] = 'diversifier expansionist'
#            portrayal["text_color"] = None


    elif type(agent) is FieldPatch:
        if agent.field_le >0.5:
            portrayal["Color"] = ["#00FF00", "#00CC00", "#009900"]
        elif agent.field_le <=0.5:
            portrayal["Color"] = ["#405c40", "#608960", "#89ac89"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1
#        portrayal["text"] = agent.field_owner_id
#        portrayal["text_color"] = 'white'


    return portrayal

########################################################
#  DEFINE VISUALIZATION COMPONENT                      #
########################################################
canvas_element = CanvasGrid(FarmerPotrayal, 30,30, 500, 500)
chart_element1 = ChartModule([{"Label": "percentage_agent_hobby", "Color": "red"},
                             {"Label": "percentage_agent_conventional", "Color": "blue"},
                             {"Label": "percentage_agent_diversifier", "Color": "black"},
                             {"Label": "percentage_agent_conventional_expansionist", "Color": "orange"},
                             {"Label": "percentage_agent_diversifier_expansionist", "Color": "gray"}],
                 canvas_height=50, canvas_width=200)

    
chart_element2 = ChartModule([{"Label": "percentage_farm_size_hobby", "Color": "red"},
                             {"Label": "percentage_farm_size_conventional", "Color": "blue"},
                             {"Label": "percentage_farm_size_diversifier", "Color": "black"},
                             {"Label": "percentage_farm_size_conventional_expansionist", "Color": "orange"},
                             {"Label": "percentage_farm_size_diversifier_expansionist", "Color": "gray"}],
                 canvas_height=50, canvas_width=200)

chart_element3 = ChartModule([{"Label": "mean_farm_size_hobby", "Color": "red"},
                             {"Label": "mean_farm_size_conventional", "Color": "blue"},
                             {"Label": "mean_farm_size_diversifier", "Color": "black"},
                             {"Label": "mean_farm_size_conventional_expansionist", "Color": "orange"},
                             {"Label": "mean_farm_size_diversifier_expansionist", "Color": "gray"}],
                 canvas_height=50, canvas_width=200)

model_params = {"scenario": UserSettableParameter('choice',\
                            name='Scenario',\
                            value ='Basic',
                            choices =['Basic','Trend','B2','A1']),
                'index_growth': UserSettableParameter('slider',\
                            name='Index Growth',\
                            value =0.01,
                            min_value =0, 
                            max_value =0.20,
                            step =0.01),
                'initial_farmers': UserSettableParameter('slider',\
                            name='Number of initial farmers',\
                            value =30,
                            min_value =10, 
                            max_value =100,
                            step =1),
                'height': UserSettableParameter('slider',\
                            name='Height of the Area',\
                            value =30,
                            min_value =5, 
                            max_value =30,
                            step =5),
                'width': UserSettableParameter('slider',\
                            name='Width of the Area',\
                            value= 30,
                            min_value =5, 
                            max_value =30,
                            step =5),
                                                     }
########################################################
#  DEFINE VISUALIZATION COMPONENT                      #
########################################################
server = ModularServer(FarmerModel, [canvas_element, chart_element2], \
                       "Farm Use Model", model_params)
server.port =1200
 # The default portal number 
server.launch()
