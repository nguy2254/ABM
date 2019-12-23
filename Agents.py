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
#          AGENTS        #
##########################  
"""
##########################
# IMPORT DATAFILE        #
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

import sys 
import os
path = 'C:\\Users\\User\\Desktop\\Quyen Nguyen\\Internship\\FarmerModel_Dutchcase\\'

sys.path.append(os.path.abspath(path))

##########################
# READ DATAFILE         #
##########################
#Define an  agent that represents a farmer 
class FieldPatch(Agent):
    ''' This is the environment class that represents a random parcel with a 
    a centroid. Each random parcel is a raster landscape
    Each agent represents the farm as a whole. It does not make decision, is immobile,
    but is updated with internalized information regards to productivity and land type'''
    def __init__(self,patch_id, pos, model,field_id, field_owner_id,field_area,\
                 field_suitability, field_le, field_le_current,\
                 field_le_potential, field_size, field_soil, \
                 field_landuse,field_ehs):
        super().__init__(patch_id, model)
        # DEFINITION OF THE FIELDS 
        self.field_owner_id=field_owner_id         # field owner 
        self.field_size=field_size             # size of the field 
        self.field_suitability=field_suitability      # suitability of the field for agriculture 
        self.field_landuse=field_landuse          # land_use type of the field
        self.field_le = field_le  # land_use type of the field
        self.field_le_current=field_le_current       # length of landscape elements in a pixel
        self.field_le_potential=field_le_potential     # perimeter of the field 
        self.field_ehs=field_ehs              # whether the field belongs to the area selected for the EHS

    #Define the step to find the associated owner of the FIELD agent 
    def find_field_owner_id(self):
        self.field_owner = None 
        if (self.field_owner_id == -1) or (self.field_owner_id == None):
            self.field_owner = None 
        else: 
            for agent in self.model.schedule.agents_by_breed[Farmer].values():
                if agent.agent_id == self.field_owner_id:
                    self.field_owner = agent

    #Define the step to calculate the overall productivity of the FARM agent 
    def find_field_distance_owner(self):
        """ Get the distance between two point, accounting for toroidal space.
        Args:
            pos_1, pos_2: Coordinate tuples for both points.
        """
        self.field_distance_owner = None
        if self.field_owner != None:
            x1, y1 = self.pos
            x2, y2 = self.field_owner.pos
            dx = np.abs(x1 - x2)
            dy = np.abs(y1 - y2)
            self.field_distance_owner = np.sqrt(dx * dx + dy * dy)

    #Define behavior of the agent in the model 
    def step(self):
        self.find_field_owner_id()
        self.find_field_distance_owner()


#Define an  agent that represents a farmer 
class Farmer(Agent):
    ''' This is the agent class that represents a farmer. Each farmer is 
    assigned to the centroid of a FARM PATCH. It has the decision making ability,
    is capable of obtian information, make farming decision and update their characteristics.
    Their farming decision is a balance between the three critical factors: 
    (1) maximize the net revenue value
    (2) get information from social network
    (3) imitate the social network '''
    def __init__(self,agent_id, pos, model, agent_type, agent_age,agent_business_type,\
                 agent_previous_transaction,agent_production,agent_production_extra,\
                 agent_national_landscape ):
        ''' FARMER(self,unique_id, pos, model, age, current_net_revenue_value, \
                 enterprise_type, likelihood_conversion, life_cycle_state, \
                 succession_ratio)'''
        super().__init__(agent_id, model)
        self.pos = pos 
        #WILLINGNESS AND ABILITY
        self.agent_type = agent_type
        #ABILITY 
        self.agent_id = agent_id #agent id
        self.agent_age =  agent_age #age of the farm head
        self.agent_production =agent_production #dsu. per ha
        self.agent_business_type =agent_business_type # agribusiness type (e.g. livestock, intensive livestock, etc)
        self.agent_previous_transaction =agent_previous_transaction # average transactions made between 2001_2005. 
        self.agent_production_extra =agent_production_extra      # extra dsu per hectare due to differences between spatial data and census data (> 2 dsu/ha)
        self.agent_new = 0
        #############################################
        # AGENTS' INITIAL CONDITIONS                #
        #############################################
        self.calculate_other_characteristics()
        self.option_agent_type()
        self.option_agent_initial()
        ## Feedback 
        self.feedback_exogenous_scenario()
        self.feedback_endogenous_landscape()
        self.feedback_internal_actions()
        self.feedback_internal_decisions()
        ## Calculate the option 
        self.farm_cessation_option()
        self.farm_expansion_option()
        self.protection_trees_option()
        ## Update the agent 
        self.update_agent()

    ############################
    # CALCULATE CHARACTERISTICS#
    ############################
    #Define the step to calculate the overall productivity of the FARM agent 
    def calculate_distance(self, other_object):
        """ Get the distance between two point, accounting for toroidal space.
        Args:
            pos_1, pos_2: Coordinate tuples for both points.
        """
        if (self == None) and (other_object == None):
            distance = None 
        else:
            x1, y1 = self.pos
            x2, y2 = other_object.pos
            dx = np.abs(x1 - x2)
            dy = np.abs(y1 - y2)
            distance = np.sqrt(dx * dx + dy * dy)
        return distance 

    #Define the step to calculate the overall productivity of the FARM agent 
    def calculate_other_characteristics(self):
        # Calculation of other agents' characteristics
        ## Calculation of other agents' characteristics
        self.agent_transactions = [self.agent_previous_transaction] * 5  # list the transactions (2001_2005) in five years 
        self.agent_farm_expansion_sum  = sum(self.agent_transactions)         # amount of land that an agent has bought/sold in the last 5 years
        self.agent_farm_expansion =  0    # create the value of land_transaction                   
        # create a list of patches that belong to each agent
        self.agent_farm_list =[field for field in \
                               self.model.schedule.get_agents_by_breed(FieldPatch).values()
                               if (field.field_owner_id == self.agent_id)]
        self.agent_farm_size = sum(field.field_size for field in self.agent_farm_list) # define the farm size
        self.agent_farm_size_previous = self.agent_farm_size  # define the previous farm size   
        self.agent_farm_size_initial=  sum(field.field_size for field in self.agent_farm_list) # define the initial farm size
        self.agent_decision_trees = random.randint(1, 10)# define randomly the year an agent participated in a policy to protect these elements
        self.agent_cessation = ""  # definition of the variable
        self.agent_tree_size = len([field for field in self.agent_farm_list if \
                                        (field.field_le ==1)] )

    #############################################
    # DEFINITION OF THE DIFFERENT PROBABILITIES #
    #############################################
    #To intialize option of the agent type 
    def option_agent_type(self):
        if self.agent_type ==1: # Hobby
            self.p_expand_type =0.01      # Combined with p_shrink
            self.p_shrink_type =0.05      # Combined with p_expand, real value = 0.12
            self.p_stop_type   =0.34      # Data based on farmers between 50_60 years old of the detailed survey 
            self.p_protect_type=0.20      # Define the probability to cut/keep/plant landscape elements
            
        elif self.agent_type == 2:# Conventional
            self.p_expand_type =0.28      # Combined with p_shrink
            self.p_shrink_type =0.04      # Combined with p_expand, real value = 0.11
            self.p_stop_type   =0.36      # Data based on farmers between 50_60 years old of the detailed survey 
            self.p_protect_type=0.32      # Define the probability to cut/keep/plant landscape elements
            
        elif self.agent_type == 3: # Diversifier
            self.p_expand_type =0.35      # Combined with p_shrink
            self.p_shrink_type =0.10      # Combined with p_expand, real value = 0.08
            self.p_stop_type   =0.32      # Data based on farmers between 50_60 years old of the detailed survey 
            self.p_protect_type=0.47      # Define the probability to cut/keep/plant landscape elements
            
        elif self.agent_type == 4:# Expansionist conventional
            self.p_expand_type =0.60      # Combined with p_shrink
            self.p_shrink_type =0.005     # Combined with p_expand, real value = 0.02
            self.p_stop_type   =0.06      # Data based on farmers between 50_60 years old of the detailed survey
            self.p_protect_type= 0.20      # Define the probability to cut/keep/plant landscape elements
            
        elif self.agent_type ==  5: # Expansionist diversifier
            self.p_expand_type =0.64      # Combined with p_shrink
            self.p_shrink_type =0.005     # Combined with p_expand, real value = 0.01
            self.p_stop_type   =0.05      # Data based on farmers between 50_60 years old of the detailed survey
            self.p_protect_type=0.47      # Define the probability to cut/keep/plant landscape elements
            
    #To intialize option of the agent
    def option_agent_initial(self):
        # Based on the previous land transactions, 
        # the initial conditions are defined in this step for:
        # Farm cessation
        self.agent_random_stop= random.uniform(0, 1)
        # Protection of landscape elements
        self.agent_random_protect = random.uniform(0, 1)
        # Farm expansion
        if self.agent_farm_expansion_sum > 0.1:
            self.agent_random_expand= random.uniform(0, self.p_expand_type + 0.1)
        
        elif self.agent_farm_expansion_sum < -0.1:
            self.agent_random_expand=  (0.9 - self.p_shrink_type)\
                + random.uniform(0, self.p_shrink_type + 0.1)
        
        else:
            self.agent_random_expand = (self.p_expand_type - 0.1)\
                +random.uniform(0,(1-self.p_shrink_type))\
                - self.p_expand_type + 0.2
                            
    #############################################
    # FEEDBACK EXOGENOUS SCENARIO               #
    #############################################
    def feedback_exogenous_scenario(self):
        if self.model.scenario == "Basic":
            self.p_exogenous_stop  = 1 # For the whole population, no influence  
            self.p_business_stop   = 0 # For the whole population, no influence
            self.p_exogenous_expand= 0 # For the whole population, no influence
            self.p_scenario_ehs    = 0                                # For the whole population, no influence
            
        elif self.model.scenario == "Trend":
            self.p_exogenous_stop  = 1.5 # For the whole population, no influence  
            self.p_business_stop   = 0 # For the whole population, no influence
            if self.agent_type <4: # Difference between the buying capacity of non_expansionist and expansionist agents
                self.p_exogenous_expand= 0.2 # non_expansionits (assumed)
            else: 
                self.p_exogenous_expand= 0.3 # expansionist (assumed)

        elif self.model.scenario == 'B2':
            self.p_exogenous_stop= 1.6 # For the whole population (assumed)
            self.p_business_stop = 0   # For the whole population, no influence
            if self.agent_business_type == 1:
                self.p_business_stop =-0.08 # Arable farms (calculated, secondary data)
            elif self.agent_business_type == 4:
                self.p_business_stop =0.15 # Dairy farms (calculated, secondary data)
            elif self.agent_business_type == 5:
                self.p_business_stop =-0.06 # Other grassland farms (calculated, secondary data)
            elif self.agent_business_type == 6:
                self.p_business_stop = 0.15 # Intensive livestock farms (calculated)
            elif self.agent_business_type == 7: 
                self.p_business_stop = 0.10 # Mixed farms (calculated, secondary data)
            if self.agent_type < 4 :# Difference between the buying capacity of non_expansionist and expansionist agents
                self.p_exogenous_expand = 0.2 # non_expansionits (assumed)
                self.p_exogenous_expand = 0.3 # expansionist (assumed) 

        elif self.model.scenario ==  "A1":
            self.p_exogenous_stop = 2 # For the whole population (assumed)
            self.p_business_stop  = 0 # For the whole population, no influence
            if self.agent_business_type == 1:
                self.p_business_stop = 0.06 # Arable farms (calculated, secondary data)
            elif self.agent_business_type ==4:
                self.p_business_stop = 0.09 # Dairy farms (calculated, secondary data)
            elif self.agent_business_type ==5:
                self.p_business_stop= -0.11 # Other grassland farms (calculated, secondary data)
            elif self.agent_business_type ==6:
                self.p_business_stop  = 0.09# Intensive livestock farms (calculated)
            elif self.agent_business_type == 7:
                self.p_business_stop =-0.04 # Mixed farms (calculated, secondary data)
            if self.agent_type < 4: # Difference between the buying capacity of non_expansionist and expansionist agents
                self.p_exogenous_expand = 0.2 # non_expansionits (assumed)
            else:
                self.p_exogenous_expand  = 0.4 # expansionist (assumed)

    #To incluence of the different scenarios in the probabilities
    def feedback_endogenous_landscape(self):
        # feedback only included in the A1 scenario. Immigrants from the urban areas can buy hobby farms (likelihood 0.5)
        # when these are located in areas surrounded more than 10% with nature (assumed).
        self.agent_random_sell  =  random.uniform(0, 1)
        # Count all patches in the study area within a radius of 1Km
        for patch in self.model.schedule.agents_by_breed[FieldPatch].values():
            patch.distance_to_agent = self.calculate_distance(patch)
        self.patches_around =len([patch for patch in\
                                 self.model.schedule.agents_by_breed[FieldPatch].values()if \
                                 (patch.field_landuse > 0) and (patch.distance_to_agent < 10)])
        # Count patches with nature in the study area within a radius of 1Km
        self.nature_around = len([patch for patch in\
                                 self.model.schedule.agents_by_breed[FieldPatch].values() if \
                                 (patch.field_landuse ==4) and (patch.distance_to_agent < 10)])
        if (self.agent_random_sell > 0.5) and (self.model.scenario == "A1") \
                    and (self.agent_farm_size < 10):
            # Calculate the propotion of surrouding nature area 
            if self.patches_around >0: 
                self.surrounding_nature = self.nature_around / self.patches_around
            else:
                self.surrounding_nature = -1
        else:
            self.surrounding_nature = -1 
        # Those with more than 10% of nature are sold to urban immigrants and probabilities are re_calculated
        if self.surrounding_nature > 0.1: 
            self.agent_random_expand = random.uniform(0, 1)
            self.agent_type= 1
            self.agent_age = 37
            self.agent_new = 1 
            self.agent_cessation = ""
            self.agent_expansion = ""
            self.agent_protection= ""
        else:
            self.agent_expansion = ""
            self.agent_protection= ""
    
    ###########################################################################
    # QUANTIFICATION OF THE INFLUENCE OF PREVIOUS DECISIONS ON FUTURE OPTIONS #
    ###########################################################################
    #To Increase of the agents' buying capacity per scenario (assumed values)
    def feedback_internal_actions(self):
        if self.model.scenario == "Basic":
            self.model.index_growth =  0
        elif self.model.scenario == "Trend":
           self.model.index_growth  = 0.1
        elif self.model.scenario == 'B2': 
            self.model.index_growth = 0.1
        elif self.model.scenario == "A1":
            self.model.index_growth= 0.3
        
        # Propabibility that an agent will increase his farm_size based on her/his current farm size 
        # and previous transactions (empirical data). Mean farm size is recalculate every time step 
        # to include the general changes of farm size of the whole population through time
        self.model.mean_farm_size = np.mean([farmer.agent_farm_size for farmer in 
                                 self.model.schedule.get_agents_by_breed(Farmer).values()])
        if self.agent_farm_size < 10: # unit ha
            if (self.agent_farm_expansion_sum <= \
                (2 * self.model.mean_farm_size * self.model.index_growth) and \
                (self.agent_farm_expansion_sum > self.model.mean_farm_size)): 
                self.p_expand_feedback = 0.86 + self.model.index_growth
            if (self.agent_farm_expansion_sum <= self.model.mean_farm_size) and \
                (self.agent_farm_expansion_sum > 2):
                self.p_expand_feedback = 0.61 + self.model.index_growth
        if self.agent_farm_size >= 10: 
            if (self.agent_farm_expansion_sum <= 2 * self.model.mean_farm_size) and \
                (self.agent_farm_expansion_sum > self.model.mean_farm_size):
                self.p_expand_feedback = 0.73 + self.model.index_growth
            if (self.agent_farm_expansion_sum <= self.model.mean_farm_size) and \
                (self.agent_farm_expansion_sum > 2):
                self.p_expand_feedback =  0.45 + self.model.index_growth

        if self.agent_farm_expansion_sum > (2 *self.model.mean_farm_size):
            self.p_expand_feedback = 0.20 + self.model.index_growth
            #this value is the same for all agents
        if self.agent_farm_expansion_sum > (4 * self.model.mean_farm_size):
            self.p_expand_feedback = (0.05 + self.model.index_growth)
            #agents cannot grow anymore
        if self.agent_farm_expansion_sum <= 2:
            self.p_expand_feedback = 1
            #previous transactions do not influence agents' options
        else:
            self.p_expand_feedback = 1
        #Propabibility that an agent will stop farming based on her/his previous transactions (empirical data)
        #The value 0.14 is half of the difference between farmers older than 50 who bought and who didn't buy any land between 2001_2005.
        if (self.agent_farm_expansion_sum < 0.1):
            self.p_stop_feedback  = 0.14
        if (self.agent_farm_expansion_sum >= 0.1):
            self.p_stop_feedback  =-0.14

    #To influence of the previous in the subsequent random number 
    def feedback_internal_decisions(self):
        # Based on the calibration of the amplitude of the curve, 0.06 was the selected value (calculated)
        self.agent_random_expand_initial = self.agent_random_expand
        self.agent_random_protect_initial = self.agent_random_protect
        self.agent_random_expand= np.random.normal(self.agent_random_expand_initial, 0.06)
        self.agent_random_protect=np.random.normal(self.agent_random_protect_initial,0.06)
        # Probabilities need to be between 0 and 1.  
        if self.agent_random_expand < 0:
            self.agent_random_expand= 0
        elif self.agent_random_expand > 1:
            self.agent_random_expand =1
        elif self.agent_random_protect < 0:
            self.agent_random_protect =0 
        elif self.agent_random_protect > 1:
            self.agent_random_protect =1

    ###########################################################################
    # CESSATION                                                               #
    ###########################################################################
    #To decide to stop farming 
    def farm_cessation_option(self):
        if self.agent_cessation == "":
            # The likelihood to stop farming depends on agent type, the influence of the exogenous processes on the agent population 
            # and the agribusiness type, and the previous land transactions 
            self.p_stop = (self.p_stop_type * (self.p_exogenous_stop) * \
                           (1 + self.p_business_stop) * (1 + self.p_stop_feedback))
            # In the B2 scenario, the probability of stop farming depends on two more variables 
            if self.model.scenario == "B2":
                # agents who live in the national landscape are less likely to stop farming (assumed)
                if self.national_landscape == 1:
                    self.p_stop = self.p_stop * 0.9
                else:
                    self.p_stop = self.p_stop * 1.1
                # diversifiers are less likely to stop in this scenario (assumed)
                if (self.agent_type == 3 or self.agent_type == 5):
                    self.p_stop = self.p_stop * 0.9
                else:
                    self.p_stop = self.p_stop *1.1
            # Probabilities need to be between 0 and 1
            if self.p_stop < 0:
                self.p_stop = 0
            if self.p_stop > 1:
                self.p_stop =1
        # if the agent has no cessation option, then their probabiity of stop is None 
        else:
            self.p_stop = 0 

    def farm_cessation_decision(self):
        if self.agent_age >50:
            # Agents older than 50 decide whether to stop or inherit their farms
            if self.agent_random_stop < self.p_stop:
                self.cessation =  "stop"
                # TRANSITIONAL RUPTURE, agents who were expansionst become non_expansinist 
                if self.agent_type == 4:
                    self.agent_type = 2
                elif self.agent_type == 5:
                    self.agent_type = 3
                # Their probabilities are recalculated, but they are 0.5 more likely to sell their farms (assumed)
                self.option_agent_type()
                self.agent_random_expand = 0.5 +  random.uniform(0, 0.5)
               # Those who don't stop will inherit their farm (no changes in agent type)
        else: 
            self.agent_cessation ="inherit" 

    def farm_cessation_action(self):
        # Depending on the scenario, \
        #the proportion of agents stopping farming each year varies. 
        #This has to be calculated per year (calculated, secondary data) 
        self.model.total_agents = self.model.schedule.get_breed_count(Farmer)
        self.model.total_agents_stop = len([agent for agent in \
                                 self.model.schedule.get_agents_by_breed(Farmer).values()\
                                 if (agent.agent_cessation == 'stop' )])
        if self.model.scenario == "Basic":
            self.model.index_stop_year = 0.1
        if self.model.scenario == "Trend":
            self.model.index_stop_year = 0.025 
#            self.model.index_stop_year = (0.025 * total_agents)/total_agents_stop
        if self.model.scenario == "B2":
            self.model.index_stop_year = 0.029 
#            self.model.index_stop_year = (0.029 * total_agents)/total_agents_stop
        if self.model.scenario == "A1":
            self.model.index_stop_year = 0.040
#            self.model.index_stop_year = (0.040 * total_agents)/total_agents_stop
        
        # only agents older than 65 years are able to inherit their farm
        if (self.agent_cessation == "inherit")  and (self.agent_age >= 65):
            self.percentage_agents =  random.uniform(0, 1)
            # Agents younger than 84 have 0.1 likelihood to inherit their farm (empirical data) 
            # and agents 84 years old inherit the farm
            if ((self.percentage_agents < 0.1) or (self.agent_age >= 84)):
                # The new agent is 37 years old (empirical data) and follows the same strategy as the predecessor
                self.agent_age =  37
                self.agent_cessation = ""

        if self.agent_cessation == "stop":
            # The process of stop farming differ between scenarios
            # Based on a random number and the expected number of agent 
            #to stop per year or when they are older than 84, agents stop farming
            self.percentage_agents =  random.uniform(0, 1)
            if (self.percentage_agents < self.model.index_stop_year) or (self.agent_age >= 84):
                if self.model.scenario == "A1":
                    # In the A1 scenario, to sell fields that are part of the EHS in order to develop nature depends on a  
                    # random number and the suitability of the field for agriculture, the lower the suitability the higher chance to be sold
                    for field in self.agent_farm_list:
                        field.random_ehs= (random.uniform(0,1) + field.field_suitability)/ 2
                        if field.random_ehs < 0.5:
                            field.field_ehs =1
                    self.fields_ehs = [field for field in self.agent_farm_list if \
                                      field.field_ehs ==1]
                    for farm in self.fields_ehs:
                        farm.field_owner_id = 9999
                        # patches sold to the nature development organisation
                        farm.field_landuse =  4
                        # new land_use nature
                    self.agent_farm_list = []
                    for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                        if field.field_owner_id == self.agent_id:
                            self.agent_farm_list.append(field)
                    
                    # Those agent without any other field will quit
                    if (self.agent_farm_list == []):
                        self.model.schedule.remove()

                    # In the A1 scenario, non_hobby agents are (0.5) likely to 
                    # sell their fields not suitable for 
                    # agriculture and outside the EHS to be converted into nature
                    self.agent_random_abandon = random.uniform(0,1)
                    if (self.agent_type > 1) and (self.agent_random_abandon > 0.5):
                        # Nature development 
                        self.fields_abandon = [field for field in self.agent_farm_list\
                                               if (field.field_suitability < 0.5)]
                        for field in self.fields_abandon:
                            field.field_owner_id= 9999
                            # patches sold to the nature development organisation
                            field.field_landuse=  4
                    # new land_use nature
                    self.agent_farm_list = []
                    for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                        if field.field_owner_id == self.agent_id:
                            self.agent_farm_list.append(field)
                            
                    # Those agent without any other field will quit
                    if (self.agent_farm_list == []):
                        self.model.schedule.remove()
                else:
                    # In the other sceanrios fields located in the EHS are abandoned 
                    self.fields_ehs = [field for field in self.agent_farm_list if \
                                      field.field_ehs ==1]
                    
                    # EHS development
                    for farm in self.fields_ehs:
                        farm.field_owner_id = 9999
                        # patches sold to the nature development organisation
                        farm.field_landuse =  4
                    
                    # new land_use nature
                    self.agent_farm_list = []
                    for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                        if field.field_owner_id == self.agent_id:
                            self.agent_farm_list.append(field)

                    # Those agent without any other field will quit
                    if (self.agent_farm_list == []):
                        self.model.schedule.remove()
                    
                    # From the land left, big farms (> 5 fields) are sold to 
                    #different buyers (assumed)
                    while len(self.agent_farm_list > 5):
                        # Selection of the buyer
                        # The buyer should be close to the seller
                        self.buyers = [agent for agent in \
                                       self.model.schedule.get_agents_by_breed(Farmer).values()\
                                       if (agent.agent_expansion == 'buy')]
                        for agent in self.buyers:
                            agent.distance_self = self.calculate_distance(agent)
                        self.buyers.sort(key=lambda \
                                                x: x.distance_self, reverse=False)
                        self.nearest_10_buyers = self.buyers[0:9]
                        # the variables to selected the buyer are reset to change 
                        # previous values
                        for agent in self.nearest_10_buyers:
                            agent.weight_size =0
                            agent.weight_distance =0
                            agent.weight_type =0
                            # the variables to select the buyer are re_calculated
                            agent.weight_random = np.random.uniform(0,1)
                            if agent.agent_farm_size > self.agent_farm_size: 
                                agent.weight_size = 0.1
                            if agent.distance_self < 20:
                                agent.weight_distance = 0.1
                            if agent.agent_type > 3:
                                agent.weight_type = 0.1
                            agent.weight_buy = (agent.weight_size + agent.weight_distance\
                                              + agent.weight_type + agent.weight_random)
                        # Selection of the buyer
                        self.nearest_10_buyers.sort(key=lambda x: x.weight_buy, reverse=True)
                        self.buyer = self.nearest_10_buyers[0]
                        # Five fields to be sold are selected  
                        self.patches_sell =random.sample(self.agent_farm_list, 5)  
                        # Land transaction
                        for patch in self.patches_sell:
                            patch.field_owner_id = self.buyer.agent_id
                        
                        # Update the land owned and land transactions by the closest buyer
                        self.buyer.agent_farm_list =[]
                        for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                            if field.field_owner_id == self.buyer.agent_id:
                                self.buyer.agent_farm_list.append(field)
                        for field in self.buyer.agent_farm_list:
                            field.find_field_owner_id()
                            field.find_field_distance_owner()
                        self.buyer.agent_expansion = "bought"

                        # Update the farm of the seller 
                        self.agent_farm_list =[]
                        for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                            if field.field_owner_id == self.agent_id:
                                self.agent_farm_list.append(field)
                        for field in self.agent_farm_list:
                            field.find_field_owner_id()
                            field.find_field_distance_owner()
                    
                    # Before agents stop farming they can sell their farm 
                    # to urban immigrants (only hobby and in the A1 scenario)
                    self.feedback_endogenous_landscape()
                    # Agents who are not new immigrants would continue selling their land
                    if self.agent_new == 0:
                        # The rest of big farms or small farms are sold to one buyer 
                        # The buyer should be close to the seller
                        # The buyer should be close to the seller
                        # The buyer should be close to the seller
                        self.buyers = [agent for agent in \
                                       self.model.schedule.get_agents_by_breed(Farmer).values()\
                                       if (agent.agent_expansion == 'buy')]
                        for agent in self.buyers:
                            agent.distance_self = self.calculate_distance(agent)
                        self.buyers.sort(key=lambda \
                                                x: x.distance_self, reverse=False)
                        self.nearest_10_buyers = self.buyers[0:9]
                        for agent in self.nearest_10_buyers:
                            # the variables to selected the buyer are reset
                            agent.weight_size =0
                            agent.weight_distance =0
                            agent.weight_type =0
                            # the variables to select the buyer are calculated
                            agent.weight_random = np.random.uniform(0,1)
                            if agent.agent_farm_size > self.agent_farm_size: 
                                agent.weight_size = 0.1
                            if agent.distance_self < 20:
                                agent.weight_distance = 0.1
                            if agent.agent_type > 3:
                                agent.weight_type = 0.1
                            agent.weight_buy = (agent.weight_size + agent.weight_distance\
                                              + agent.weight_type + agent.weight_random)
                        # Selection of the buyer
                        self.buyers.sort(key=lambda \
                                                x: x.weight_buy, reverse=True)
                        if self.buyers !=[]:
                            self.nearest_10_buyers = self.buyers[0:9]
                            self.buyer = self.nearest_10_buyers[0]
                            # Land transaction
                            for field in self.agent_farm_list:
                                field.field_owner_id = self.buyer.agent_id
    
                            # Update the land owned and land transactions by the closest buyer
                            self.buyer.agent_farm_list =[]
                            for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                                if field.field_owner_id == self.buyer.agent_id:
                                    self.buyer.agent_farm_list.append(field)
                            for field in self.buyer.agent_farm_list:
                                field.find_field_owner_id()
                                field.find_field_distance_owner()
                            self.buyer.agent_expansion = "bought"

                        # Update the farm of the seller 
                        self.agent_farm_list =[]
                        for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                            if field.field_owner_id == self.agent_id:
                                self.agent_farm_list.append(field)
                        for field in self.agent_farm_list:
                            field.find_field_owner_id()
                            field.find_field_distance_owner()
                    
                    # Those agent without any other field will quit
                    if (self.agent_farm_list == []):
                        self.model.schedule.remove()

    ###########################################################################
    # EXPANSION/SHRINKAGE                                                     #
    ###########################################################################
    #To decide to expand farming 
    def farm_expansion_option(self):
        # The likelihood to buy/sell land depends on the agent type, the previous land transactions and 
        # the exogenous processes of each scenario
        self.p_expand =  (self.p_expand_type * self.p_expand_feedback * \
                          (1 + self.p_exogenous_expand)) 
        # In the B2 scanerio, because the active development of the EHS, agents with fields in the EHS are 100% 
        # more likely to sell that parcel (assumed)
        if  self.model.scenario == "B2":
            self.count_field_le = 0 
            for farm in self.agent_farm_list:
                if farm.field_ehs ==1: 
                    self.count_field_ehs +=1
            if self.count_field_ehs >=1:
                self.p_scenario_ehs= 1
                self.p_scenario_ehs= 0
        else:
            self.p_scenario_ehs = 0
        self.p_shrink = (self.p_shrink_type * (1 + self.p_scenario_ehs))  
        # Probabilities need to be between 0 and 1
        if self.p_expand < 0:
            self.p_expand = 0 
        elif self.p_expand > 1:
            self.p_expand = 1
        elif self.p_shrink < 0:
            self.p_shrink = 0
        elif self.p_shrink > 1:
            self.p_shrink = 1

    def farm_expansion_decision(self):
        # Agents who will stop are quite unlikely to grow (only 7%), 
        # therefore it's assumed they don't grow (empirical data)
        # The decision of buying land depends on three main factors:
        if ((self.p_expand > self.agent_random_expand) and \
                # The agent wants to expand AND
                (self.agent_cessation != "stop") and\
                # the agent is not planning to stop farming AND
                (self.agent_farm_expansion_sum > -1)):
            # the agent hasn't sold any land > 1ha. in the last five years.
            self.agent_expansion = "buy"
            self.agent_expansion = "stable"
            
        # The decision of selling land depends on three factors 
        if (((1 - self.p_shrink) < self.agent_random_expand) and\
                # The agent wants to sell AND
               (len(self.agent_farm_list) > 1) and\
               # the agent has more than 1 field AND
               (self.agent_farm_expansion_sum < 1)):
            # the agent hasn't bought any land > 1ha. in the last five years.
            self.agent_expansion= "sell"

    def farm_expansion_action(self):
        # Transaction when an agent leases/sell only a field. 
        # This process also depends on the scenario.
        if self.agent_expansion == "sell":
            # In the B2 scenario, fields in the EHS are sold. 
            # If agents don't have one in the EHS, they sell the farthest one 
            if self.model.scenario == "B2":
                self.fields_ehs = [field for field in self.agent_farm_list if \
                                      field.field_ehs ==1]
                if self.fields_ehs != []:
                    self.field_sell = random.sample(self.fields_ehs,1)
                else:
                    self.agent_farm_list.sort(key=lambda \
                                                x: x.field_distance_owner, reverse=True)
                    
                    self.field_sell =self.agent_farm_list[0]
            else:
                self.agent_farm_list.sort(key=lambda \
                                                x: x.field_distance_owner, reverse=True)
                self.field_sell =self.agent_farm_list[0]
                # In the Basic, Trend and B2 scenarios,\
                #fields that are in the EHS are developed into nature.
                # In the A1 scenrio, only those located 
                #in the EHS with low suitability are converted to nature.
                if (((self.model.scenario != "A1") and\
                    (self.field_sell.field_ehs== 1)) or\
                    ((self.model.scenario == 'A1') and\
                     (self.field_sell.field_ehs ==-1 ) and \
                     (self.filed_sell.field_suitability < 0.5))):
                    # Nature development  
                    self.field_sell.field_owner_id = 9999
                    self.field_sell.field_landuse =4
                    self.agent_farm_list=[]
                    for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                        if field.field_owner_id == self.agent_id:
                            self.agent_farm_list.append(field)
                    self.field_sell = []
                    # Update the farm of the seller
                    self.agent_farm_list =[]
                    for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                        if field.field_owner_id == self.agent_id:
                            self.agent_farm_list.append(field)
                    for field in self.agent_farm_list:
                        field.find_field_owner_id()
                        field.find_field_distance_owner()

                
                else:
                    # Individual fields are sold to the closest buyer 
                    self.buyers = [agent for agent in \
                                       self.model.schedule.get_agents_by_breed(Farmer).values()\
                                       if (agent.agent_expansion == 'buy')]
                    for agent in self.buyers:
                        agent.distance_self = self.calculate_distance(agent)
                    self.buyers.sort(key=lambda x: x.distance_self, reverse=False)
                    if self.buyers != []:
                        self.closest_buyer = self.buyers[0]
                        # Land transaction
                        for field in [self.field_sell]:
                                field.field_owner_id = self.closest_buyer.agent_id
                        # Update the land owned and land transactions by the closest buyer
                        self.closest_buyer.agent_farm_list =[]
                        for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                            if field.field_owner_id == self.closest_buyer.agent_id:
                                self.closest_.agent_farm_list.append(field)
                        for field in self.closest_buyer.agent_farm_list:
                            field.find_field_owner_id()
                            field.find_field_distance_owner()
                        self.closet_buyer.agent_expansion = "bought"
                    
                # Update the farm of the seller
                self.agent_farm_list =[]
                for field in self.model.schedule.get_agents_by_breed(FieldPatch).values():
                    if field.field_owner_id == self.agent_id:
                        self.agent_farm_list.append(field)
                for field in self.agent_farm_list:
                    field.find_field_owner_id()
                    field.find_field_distance_owner()
                self.agent_expansion = "sold"
                
                #Those agent without any other field will quit
                if (self.agent_farm_list == []):
                    self.model.schedule.remove()

    ###########################################################################
    # PROTECT TREE                                                            #
    ###########################################################################
    #To decide to protect tree 
    def protection_trees_option(self):
        # The likelihood to protect/cut landscape elements only depends on the type
        self.p_protect = self.p_protect_type 


    def protection_trees_decision(self):
        # The decision of cutting and planting landscape elements depends on the scenario.
        if self.model.scenario == "A1":
            # period for this kind protection programmes
            if self.agent_decision_trees >= 6:
                # The cutting of landscapa elements depends on whether there is an element in agents' field and
                # on the suitability of the field
                self.field_cut=[farm for farm in self.agent_farm_list if \
                                (farm.field_le ==1) and (farm.field_suitability > 0.5)]
                # Those who have landscdape elements in their fields, 
                # who want to cut them and who are not hobby 
                # agents would cut them. The other agents would keep them.
                if ((self.field_cut != []) and (self.p_protect < self.agent_random_protect)\
                                and (self.agent_type != 1)):
                    self.protection = "cut"
                else:
                    self.protection = "keep"
                # After they take a decision, the new decision time is 0
                self.agent_decision_trees = 0
        if self.model.scenario == "B2":
            # Agents can only take decisions 6 years after 
            # their last decisions, which is the normal 
            # period for this kind protection programmes
            if self.agent_decision_trees >= 6:
                # The decision to cut landscape elements depend on three factors
                if ((self.trees == 1) and  \
                        # the agent has these elements in the farm
                        (self.p_protect * 1.5 < self.agent_random_protect) and  \
                        # the agent wants to cut them (assumed)
                        (self.agent_type != 1)): 
                        # the agent is not a hobby agent
                    self.protection = "cut"
                    self.protection = "keep"
            # The decision to plant new landscdape elements depend on two factors 
            if ((self.p_protect > self.agent_random_protect) and\
                    # The agent wants to plant them
                    (self._agent_type != 1)): 
                    # The agent is not a hobby agent
                    self.protection = "plant"
            # After they take a decision, the new decision time is 0
            self.agent_decision_trees = 0 

    def protection_trees_action(self):
        # The process to cut or plant landscape elements depend on the scenario
        if self.model.scenario == "A1":
            if self.agent_protection == "cut":
                # Patches where landscape can be cut are chosen based on three factors
                self.patches_cut = [field for field in self.agent_farm_list if \
                                     (field.field_owner_id == self.agent_id) and\
                                    (field.field_le == 1) and\
                                    # the field has landscape elements
                                    (field.field_suitability > 0.5)\
                                    # the field is suitable for agriculture
                                    ]
                # For those you want and who have fields with landspcape elements cut them                
                if self.patches_cut != None:
                    # Change the landuse type of that field
                    self.patches_cut.field_le = 0
                    self.patches_cut.field_le_current = 0
                    self.protection = "done"
                    self.agent_cut = 1
        
        elif self.model.scenario == "B2":
            if self.agent_protection == "cut":
                # Patches where landscape can be cut are chosen based on three factors
                self.patches_cut =  [field for field in self.agent_farm_list if \
                                     (field.field_owner_id == self.agent_id) and\
                                    (field.field_le == 1) and\
                                    # the field has landscape elements
                                    (field.field_suitability > 0.7)\
                                    # the field is highly suitable for agriculture
                                    ]
                # For those you want and who have fields with landspcape
                #elements cut them
                if self.patches_cut != None:
                    # Change the landuse type of that field
                    self.patches_cut.field_le = 0
                    self.patches_cut.field_le_current = 0
                    self.protection = "done"
                    self.agent_cut = 1
                    
                if self.agent_protection ==  "plant":
                    # Patches where landscape elements can be
                    # planted are chosen based on three factors 
                    self.fields_plant = [field for field in self.agent_farm_list \
                                         if (field.field_owner_id ==self.agent_id) and\
                                     # the agents owns the field\
                                     (field.field_suitability < 0.6) and\
                                     # the field is not highly suitable for agriculture
                                     (field.field_le_current / field.field_le_potential < 0.25)]
                                     # the field doesn't have many landscape elements
                                     
                    # For those you want and who have fields with 
                    # landspcape elements to be planted
                    if self.fields_plant != []:
                        # Selection of the field 
                        self.field_plant = random.sample(self.fields_plant)
                    # Change the amount of landscape elements in the field
                    self.field_plant.field_le = 1
                    self.field_plant.field_le_current = (self.field_plant.field_le_current +\
                                                         self.field_plant.field_le_potential / 4)
                    self.agent_protection = "done"
                    self.agent_plant = 1

    #############################################
    # UPDATE AGENT AFTER EACH TRANSACTION       #
    #############################################
    #To update the charactersitics of a farm agent at each time step 
    def update_agent(self):
        # Define the production of each agent
        if self.agent_farm_size < 1:
            self.agent_production_scale = self.agent_production + self.agent_production_extra  # This avoids the numerical problems of agents with very small farms.
        else:
            self.agent_production_scale= ((self.agent_farm_size * self.agent_production) + self.agent_production_extra)
            
        # Define the production scale of each agent
        if self.agent_production_scale <= 20:
            self.agent_production_class =  "hobby"
        elif self.agent_production_scale > 20:
            self.agent_production_class ="small"
        elif self.agent_production_scale > 50:
            self.agent_production_class = "medium"
        elif self.agent_production_scale > 100:
            self.agent_production_class ="large"
            
        # Define the farm to which a patch belongs
        self.patch_farm_area =  self.agent_farm_size
        for farm in self.agent_farm_list:
            farm.find_field_owner_id()
            farm.find_field_distance_owner() # define the distance of each patch to the house
            farm.patch_farm_size= self.patch_farm_area       # define the size of the farm to which a patch belongs
            
        # Define whether an agent has fields that are with landscape elements or not
        self.count_field_le = len([farm for farm in self.agent_farm_list if \
                                   (farm.field_le ==1)])
        if self.count_field_le >=1:
            self.agent_trees = 1
        else:
            self.agent_trees = 0
            
        # FEEDBACK BETWEEN AGENT TYPES _ HOBBY VS. OTHERS
        # Update the agent type of new hobby agents
        if (self.agent_type != 1) and (self.agent_production_scale <= 20):
            self.agent_type = 1
            self.option_agent_type()

        # Changes in agent types from hobby to conventional diversifier agent 
        if (self.agent_type == 1) and (self.agent_production_scale > 20) \
                and (self.agent_new == 0):
            self.agent_type = 2
            self.option_agent_type()
        
        # FEEDBACK BETWEEN AGENT TYPES _ HOBBY VS. OTHERS
        self.agent_age += 1 # agents get older
        self.agent_agent_decision_trees=self.agent_decision_trees + 1 
        # agents' participation in policies get another year
        
    #To update the charactersitics of a farm agent at each time step 
    def update_agent_transactions(self):
        #Update the list of action_agent_farm_expansion
        self.agent_farm_size =sum(field.field_size for field in self.agent_farm_list)
        self.agent_tree_size = len([field for field in self.agent_farm_list if \
                                        (field.field_le ==1)] )
        self.agent_farm_size_previous = self.agent_farm_size #define the current farm size as previous for the subsequent year
        self.agent_farm_expansion  = self.agent_farm_size -self.agent_farm_size_previous #whether an agent has previously expanded or decreased land
        self.agent_transactions.remove(self.agent_transactions[0])#delete the land transaction of the first year 
        self.agent_transactions.append(self.agent_farm_expansion) #add the land transactions of the current year
        self.agent_farm_expansion_sum  =sum(self.agent_transactions)#update the amount of land that an agent has bought/sold in the last 5 years
        self.patch_farm_area = self.agent_farm_size #define a variable that can be used for defining the farm to which a patch belongs
        for farm in self.agent_farm_list:
            farm.patch_farm_size = self.patch_farm_area

    #############################################
    # AGENTS' INITIAL CONDITIONS                #
    #############################################
    # Calculation of other agents' characteristics
    def step(self):
        ## Calculation of other agents' characteristics
        ## Feedback 
        self.feedback_internal_actions()
        self.feedback_internal_decisions()
        ## Cessation 
        self.farm_cessation_option()
        self.farm_cessation_decision()
        self.farm_cessation_action()

        ###Expansion/Shrinkage 
        self.farm_expansion_option()
        self.farm_expansion_decision()
        self.farm_expansion_action()

        ###Protect tree 
        self.protection_trees_option()
        self.protection_trees_decision()
        self.protection_trees_action()
       
        ## Update the agent 
        self.update_agent()
        self.update_agent_transactions()
