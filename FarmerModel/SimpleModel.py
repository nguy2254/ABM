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
#          MODEL         #
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

##########################
# CREATE FARMER MODEL    #
##########################
#Create Schedule 
class FarmerModel(Model):
    '''
    Create a FARM LANDUSE MODEL with the given parameters.
        Args:
            verbose: Define if needs to print out main results after each step
            fake_data: Define if the model will generate fake data from predefined 
            parameter. 
            real_data: If fake_data = False, the real data should be provided. It is 
            a vector that includes two datasets, 1st one is farmer data and 2nd one is 
            land use data. If fake_data, then this parameter is [None, None].
            height: If fake_data =True only. Define the height of the entire area 
            width: If fake_data =True only. Define the width of the entire area 
            initial_farmers: If fake_data =True only. Define the number of the farmers
            scenario: 4 scenarios: 'Base', 'Trend', 'A1', 'B2',
            index_growth: the baseline growth of the commodity price in the model 
    '''
    description = 'A model for simulating land use conversion'

    def __init__(self, verbose=True,\
                 fake_data = True, 
                 real_data = [None,None],
                 height = 30, \
                 width =30, \
                 initial_farmers =30,\
                 scenario = 'Basic', \
                 index_growth = 0.01):
        '''
        Create a FARM LANDUSE MODEL with the given parameters.
        Args:
            verbose: Define if needs to print out main results after each step
            fake_data: Define if the model will generate fake data from predefined 
            parameter. 
            real_data: If fake_data = False, the real data should be provided. It is 
            a vector that includes two datasets, 1st one is farmer data and 2nd one is 
            land use data. If fake_data, then this parameter is [None, None].
            height: If fake_data =True only. Define the height of the entire area 
            width: If fake_data =True only. Define the width of the entire area 
            initial_farmers: If fake_data =True only. Define the number of the farmers
            scenario: 4 scenarios: 'Base', 'Trend', 'A1', 'B2',
            index_growth: the baseline growth of the commodity price in the model 
        '''
        super().__init__()

        #Set additional parameters 
        self.scenario = scenario
        self.index_growth = index_growth 
        self.stepcounter=0
        #Real data only needed if fake_data = False 
        self.fake_data= fake_data 
        self.real_data = real_data
        #Result
        self.result =[]
        ##########################
        # CREATE FAKE DATA       #
        ##########################
        if self.fake_data == True:
            self.height= height
            self.width = width
            self.initial_farmers = initial_farmers
            self.area = self.height * self.width 
            if self.initial_farmers > self.height * self.width:
                print('Error: The number of farmers are bigger than the number of farms')
            else: 
                #Create agent data 
                col_names =  ['agent_id', 'agent_x', 'agent_y',\
                              'agent_type', 'agent_business',\
                              'agent_age', 'agent_nlandscape', \
                              'agent_product', 'agent_product_extra',\
                              'agent_trans']
                final_agent_data= pd.DataFrame(columns = col_names)
                final_agent_data['agent_id'] = np.arange(1,self.initial_farmers+1,1)
                final_agent_data['agent_x'] = np.random.randint(0,self.height,size=(self.initial_farmers,))
                final_agent_data['agent_y'] = np.random.randint(0,self.width,size=(self.initial_farmers,))
                final_agent_data['agent_type'] = np.random.randint(1,6,size=(self.initial_farmers,))
                final_agent_data['agent_business'] = np.random.randint(1,6,size=(self.initial_farmers,))
                final_agent_data['agent_age'] = np.random.randint(37,100,size=(self.initial_farmers,))
                final_agent_data['agent_nlandscape'] =np.random.uniform(0,.1,size=(self.initial_farmers,))
                final_agent_data['agent_product'] = np.random.uniform(0,100,size=(self.initial_farmers,))
                final_agent_data['agent_product_extra'] = np.random.uniform(0,1000,size=(self.initial_farmers,))
                final_agent_data['agent_trans'] = np.random.uniform(-9,63,size=(self.initial_farmers,))
                
                #Create field data 
                self.fields_suitability= pd.DataFrame(np.random.uniform(0,1,size=(self.height,self.width)))
                self.fields_area =pd.DataFrame(np.random.uniform(1,5,size=(self.width,self.width)))
                self.fields_ehs = pd.DataFrame(np.random.randint(0,2,size=(self.width,self.width)))
                self.fields_id = pd.DataFrame(np.arange(1,self.area+1,1).reshape(self.height,self.width))
                self.fields_le = pd.DataFrame(np.random.uniform(0,1,size=(self.height,self.width)))
                self.fields_le_current = pd.DataFrame(np.random.uniform(0,1,size=(self.height,self.width)))
                self.fields_le_potential =pd.DataFrame(np.random.uniform(0,1,size=(self.height,self.width)))
                self.fields_owner =  pd.DataFrame(np.random.choice(final_agent_data['agent_id'],\
                                                               size=(self.height,self.width)))
                self.fields_size = pd.DataFrame(np.random.uniform(1,5,size=(self.height,self.width)))
                self.fields_soil = pd.DataFrame(np.random.uniform(0,1,size=(self.height,self.width)))
                self.fields_landuse = pd.DataFrame(np.random.choice([0,6,5,4],\
                                                               p= [0.3,0.3,0.35,0.05],\
                                                               size=(self.height,self.width)))
                #Combine field data into one single dictionary that 
                # include mutliple data tabel
                self.farmer_data = final_agent_data
                self.final_fields_data_dictionary= {'fields_suitability': self.fields_suitability,\
                                              'fields_area': self.fields_area,
                                              'fields_ehs': self.fields_ehs,
                                              'fields_id': self.fields_id,
                                              'fields_le': self.fields_le,
                                              'fields_le_current':self.fields_le_current,
                                              'fields_le_potential':self.fields_le_potential,
                                              'fields_owner': self.fields_owner,
                                              'fields_size': self.fields_size,
                                              'fields_soil': self.fields_soil,
                                              'fields_landuse':self.fields_landuse}
        if self.fake_data == False:
            self.farmer_data = real_data[0]
            self.final_fields_data_dictionary =real_data[1]
            # Add farmer agent and field 
            self.fields_suitability = self.final_fields_data_dictionary.get('fields_suitability')
            self.fields_area = self.final_fields_data_dictionary.get('fields_area')
            self.fields_ehs = self.final_fields_data_dictionary.get('fields_ehs')
            self.fields_id = self.final_fields_data_dictionary.get('fields_id')
            self.fields_le = self.final_fields_data_dictionary.get('fields_le')
            self.fields_le_current = self.final_fields_data_dictionary.get('fields_le_current')
            self.fields_le_potential = self.final_fields_data_dictionary.get('fields_le_potential')
            self.fields_owner = self.final_fields_data_dictionary.get('fields_owner')
            self.fields_size = self.final_fields_data_dictionary.get('fields_size')
            self.fields_ehs = self.final_fields_data_dictionary.get('fields_ehs')
            self.fields_landuse = self.final_fields_data_dictionary.get('fields_landuse')
            self.fields_soil = self.final_fields_data_dictionary.get('fields_soil')
            self.height, self.width = self.fields_id.shape[1],self.fields_id.shape[0]
            self.initial_farmers = self.farmer_data.shape[0]

        ##########################
        # READ DATAFILE         #
        ##########################
        #Set up the height and weight of the model equivalent to the size of farm patch
        self.grid = MultiGrid(self.height, self.width, torus=True)
        self.schedule = RandomActivationByBreed(self)
        # Create grass patches
        for agent, x, y in self.grid.coord_iter():
            field_suitability = self.fields_suitability[x][y]
            field_area = self.fields_area[x][y]
            field_ehs = self.fields_ehs[x][y]
            field_id = self.fields_id[x][y]
            field_le= self.fields_le[x][y]
            field_le_current = self.fields_le_current[x][y]
            field_le_potential = self.fields_le_potential[x][y]
            field_owner_id = self.fields_owner[x][y]
            field_size = self.fields_size[x][y]
            field_ehs = self.fields_ehs[x][y]
            field_landuse = self.fields_landuse[x][y]
            field_soil = self.fields_soil[x][y]
            if field_id !=None or field_id !=0:
                fieldpatch = FieldPatch(self.next_id(), (x, y), self,
                                   field_id,field_owner_id,field_area,\
                                   field_suitability, field_le, field_le_current, \
                                   field_le_potential, field_size, field_soil, \
                                   field_landuse,field_ehs)
                self.grid.place_agent( fieldpatch,(x, y))
                self.schedule.add(fieldpatch)
        print('Done for .....Field')

        # Read data file for the farmers 
        self.agent_id_list = self.farmer_data['agent_id'].unique()
        for agent_id in self.agent_id_list:
            x = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_x'].values[0]
            y = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_y'].values[0]
            agent_type = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_type'].values[0]
            agent_business_type = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_business'].values[0]
            agent_age = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_age'].values[0]
            agent_national_landscape = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_nlandscape'].values[0]
            agent_production = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_product'].values[0]
            agent_production_extra = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_product_extra'].values[0]
            agent_previous_transaction = self.farmer_data[self.farmer_data['agent_id']==agent_id]['agent_trans'].values[0]
            farmer= Farmer(agent_id, (x, y), self, agent_type,\
                           agent_age,agent_business_type,agent_previous_transaction,
                           agent_production,agent_production_extra,
                           agent_national_landscape)
            if farmer != None:
                self.grid.place_agent(farmer, (x, y))
                self.schedule.add(farmer)
        print('Done for....Agent')

        self.running = True
        self.calculate_data()
        self.datacollector = DataCollector(
                {"percentage_agent_hobby": lambda m: m.percentage_agent_hobby,
                 "mean_land_use": lambda m: m.mean_land_use,
                 "percentage_agent_conventional": lambda m: m.percentage_agent_conventional,
                 "percentage_agent_diversifier": lambda m: m.percentage_agent_diversifier,
                 "percentage_agent_conventional_expansionist": lambda m: m.percentage_agent_conventional_expansionist,
                 "percentage_agent_diversifier_expansionist": lambda m: m.percentage_agent_diversifier_expansionist,
                 "percentage_farm_size_hobby":lambda m: m.percentage_farm_size_hobby,
                 "percentage_farm_size_conventional": lambda m: m.percentage_farm_size_conventional,
                 "percentage_farm_size_diversifier": lambda m: m.percentage_farm_size_diversifier,
                 "percentage_farm_size_conventional_expansionist": lambda m: m.percentage_farm_size_conventional_expansionist,
                 "percentage_farm_size_diversifier_expansionist": lambda m: m.percentage_farm_size_diversifier_expansionist,
                 "mean_farm_size_hobby":lambda m: m.mean_farm_size_hobby,
                 "mean_farm_size_conventional":lambda m: m.mean_farm_size_conventional,
                 "mean_farm_size_diversifier":lambda m: m.mean_farm_size_diversifier,
                 "mean_farm_size_conventional_expansionist":lambda m: m.mean_farm_size_conventional_expansionist,
                 "mean_farm_size_diversifier_expansionist":lambda m: m.mean_farm_size_diversifier_expansionist,
                 "percentage_agent_tree_hobby": lambda m: m.percentage_agent_tree_hobby,
                 "percentage_agent_tree_conventional": lambda m: m.percentage_agent_tree_conventional,
                 "percentage_agent_tree_diversifier": lambda m: m.percentage_agent_tree_diversifier,
                 "percentage_agent_tree_conventional_expansionist": lambda m: m.percentage_agent_tree_conventional_expansionist,
                 "percentage_agent_tree_diversifier_expansionist": lambda m: m.percentage_agent_tree_diversifier_expansionist,
                 "nature": lambda m: m.nature
                 })
        self.datacollector.collect(self)

    def calculate_data(self):
        #Generic calculation
        self.total_agents = self.schedule.get_breed_count(Farmer)
        self.total_farm_size = sum([field.field_size for field in \
                                 self.schedule.get_agents_by_breed(FieldPatch).values()])
        self.mean_land_use = np.mean([field.field_le for field in \
                                 self.schedule.get_agents_by_breed(FieldPatch).values()])
        self.agent_hobby = len([farmer for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 1)]) 
        self.agent_conventional = len([farmer for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 2)])
        self.agent_diversifier= len([farmer for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 3)])
        self.agent_conventional_expansionist= len([farmer for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 4)])
        self.agent_diversifier_expansionist = len([farmer for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 5)])
        #Percentage of farmers
        self.percentage_agent_hobby = self.agent_hobby /self.total_agents
        self.percentage_agent_conventional = self.agent_conventional/self.total_agents
        self.percentage_agent_diversifier= self.agent_diversifier/self.total_agents
        self.percentage_agent_conventional_expansionist= self.agent_conventional_expansionist/self.total_agents
        self.percentage_agent_diversifier_expansionist = self.agent_diversifier_expansionist/self.total_agents

        #Percentage of area 
        self.percentage_farm_size_hobby = sum([farmer.agent_farm_size for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 1)])/self.total_farm_size
        self.percentage_farm_size_conventional = sum([farmer.agent_farm_size for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 2)])/self.total_farm_size
        self.percentage_farm_size_diversifier= sum([farmer.agent_farm_size for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 3)])/self.total_farm_size
        self.percentage_farm_size_conventional_expansionist= sum([farmer.agent_farm_size for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 4)])/self.total_farm_size
        self.percentage_farm_size_diversifier_expansionist = sum([farmer.agent_farm_size for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 5)])/self.total_farm_size

        #Percentage of area 
        self.mean_farm_size_hobby = np.mean([farmer.agent_farm_size for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 1)])
        self.mean_farm_size_conventional = np.mean([farmer.agent_farm_size for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 2)])
        self.mean_farm_size_diversifier= np.mean([farmer.agent_farm_size for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 3)])
        self.mean_farm_size_conventional_expansionist= np.mean([farmer.agent_farm_size for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 4)])
        self.mean_farm_size_diversifier_expansionist = np.mean([farmer.agent_farm_size for farmer in 
                                 self.schedule.get_agents_by_breed(Farmer).values()
                                 if (farmer.agent_type == 5)])
        #Nature 
        self.nature = sum([field.field_size for field in \
                                 self.schedule.get_agents_by_breed(FieldPatch).values()\
                                 if (field.field_owner_id == 9999)])/ \
                                  sum([field.field_size for field in \
                                 self.schedule.get_agents_by_breed(FieldPatch).values()])
        #Total farmers with natural landscape 
        if self.agent_hobby!=0:
            self.percentage_agent_tree_hobby = len([farmer for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 1) and \
                                 (farmer.agent_tree_size>=1)])/self.agent_hobby
        else:
            self.percentage_agent_tree_hobby =0 
        if self.agent_conventional!=0:
            self.percentage_agent_tree_conventional = len([farmer for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 2) and \
                                 (farmer.agent_tree_size>=1)])/self.agent_conventional
        else:
            self.percentage_agent_tree_conventional =0 
        if self.agent_diversifier !=0:
            self.percentage_agent_tree_diversifier = len([farmer for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 3) and \
                                 (farmer.agent_tree_size>=1)])/self.agent_diversifier
        else:
            self.percentage_agent_tree_diversifier =0 
        if self.agent_conventional_expansionist !=0:
            self.percentage_agent_tree_conventional_expansionist = len([farmer for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 4) and \
                                 (farmer.agent_tree_size>=1)])/self.agent_conventional_expansionist
        else:
            self.percentage_agent_tree_conventional_expansionist =0 
        if self.agent_diversifier_expansionist !=0:
            self.percentage_agent_tree_diversifier_expansionist = len([farmer for farmer in \
                                 self.schedule.get_agents_by_breed(Farmer).values()\
                                 if (farmer.agent_type == 5) and \
                                 (farmer.agent_tree_size>=1)])/self.agent_diversifier_expansionist
        else:
            self.percentage_agent_tree_diversifier_expansionist =0
            
    def step(self):
        self.calculate_data()
        self.datacollector = DataCollector(
                {"percentage_agent_hobby": lambda m: m.percentage_agent_hobby,
                 "mean_land_use": lambda m: m.mean_land_use,
                 "percentage_agent_conventional": lambda m: m.percentage_agent_conventional,
                 "percentage_agent_diversifier": lambda m: m.percentage_agent_diversifier,
                 "percentage_agent_conventional_expansionist": lambda m: m.percentage_agent_conventional_expansionist,
                 "percentage_agent_diversifier_expansionist": lambda m: m.percentage_agent_diversifier_expansionist,
                 "percentage_farm_size_hobby":lambda m: m.percentage_farm_size_hobby,
                 "percentage_farm_size_conventional": lambda m: m.percentage_farm_size_conventional,
                 "percentage_farm_size_diversifier": lambda m: m.percentage_farm_size_diversifier,
                 "percentage_farm_size_conventional_expansionist": lambda m: m.percentage_farm_size_conventional_expansionist,
                 "percentage_farm_size_diversifier_expansionist": lambda m: m.percentage_farm_size_diversifier_expansionist,
                 "mean_farm_size_hobby":lambda m: m.mean_farm_size_hobby,
                 "mean_farm_size_conventional":lambda m: m.mean_farm_size_conventional,
                 "mean_farm_size_diversifier":lambda m: m.mean_farm_size_diversifier,
                 "mean_farm_size_conventional_expansionist":lambda m: m.mean_farm_size_conventional_expansionist,
                 "mean_farm_size_diversifier_expansionist":lambda m: m.mean_farm_size_diversifier_expansionist,
                 "percentage_agent_tree_hobby": lambda m: m.percentage_agent_tree_hobby,
                 "percentage_agent_tree_conventional": lambda m: m.percentage_agent_tree_conventional,
                 "percentage_agent_tree_diversifier": lambda m: m.percentage_agent_tree_diversifier,
                 "percentage_agent_tree_conventional_expansionist": lambda m: m.percentage_agent_tree_conventional_expansionist,
                 "percentage_agent_tree_diversifier_expansionist": lambda m: m.percentage_agent_tree_diversifier_expansionist,
                 "nature": lambda m: m.nature
                 })
        self.datacollector.collect(self)
        self.schedule.step()
        self.stepcounter +=1
        
    def run_model(self, step_count=15):
        self.datacollector.collect(self)
        print("Initial mean_land_use", self.mean_land_use)
        print("Initial percentage_agent_hobby", self.percentage_agent_hobby)
        print("Initial percentage_agent_conventional", self.percentage_agent_conventional)
        print("Initial percentage_agent_diversifier", self.percentage_agent_diversifier)
        print("Initial percentage_agent_conventional_expansionist", self.percentage_agent_conventional_expansionist)
        print("Initial percentage_agent_diversifier_expansionist", self.percentage_agent_diversifier_expansionist)
        print("Initial percentage_farm_size_hobby",  self.percentage_farm_size_hobby)
        print("Initial percentage_farm_size_conventional", self.percentage_farm_size_conventional)
        print("Initial percentage_farm_size_diversifier", self.percentage_farm_size_diversifier)
        print("Initial percentage_farm_size_conventional_expansionist", self.percentage_farm_size_conventional_expansionist)
        print("Initial percentage_farm_size_diversifier_expansionist", self.percentage_farm_size_diversifier_expansionist)
        print("Initial mean_farm_size_hobby", self.mean_farm_size_hobby)
        print("Initial mean_farm_size_conventional", self.mean_farm_size_conventional)
        print("Initial mean_farm_size_diversifier", self.mean_farm_size_diversifier)
        print("Initial mean_farm_size_conventional_expansionist", self.mean_farm_size_conventional_expansionist)
        print("Initial mean_farm_size_diversifier_expansionist", self.mean_farm_size_diversifier_expansionist)
        print("Initial percentage_agent_tree_hobby", self.percentage_agent_tree_hobby)
        print("Initial percentage_agent_tree_conventional", self.percentage_agent_tree_conventional)
        print("Initial percentage_agent_tree_diversifier", self.percentage_agent_tree_diversifier)
        print("Initial percentage_agent_tree_conventional_expansionist", self.percentage_agent_tree_conventional_expansionist)
        print("Initial percentage_agent_tree_diversifier_expansionist", self.percentage_agent_tree_diversifier_expansionist)
        print("Initial nature", self.nature)
        self.result.append(self.datacollector.get_model_vars_dataframe().values[0])
        for i in range(step_count):
            self.step()
            print('')
            print('Step count', i)
            print("This step mean_land_use", self.mean_land_use)
            print("This step percentage_agent_hobby", self.percentage_agent_hobby)
            print("This step percentage_agent_conventional", self.percentage_agent_conventional)
            print("This step percentage_agent_diversifier", self.percentage_agent_diversifier)
            print("This step percentage_agent_conventional_expansionist", self.percentage_agent_conventional_expansionist)
            print("This step percentage_agent_diversifier_expansionist", self.percentage_agent_diversifier_expansionist)
            print("This step percentage_farm_size_hobby",  self.percentage_farm_size_hobby)
            print("This step percentage_farm_size_conventional", self.percentage_farm_size_conventional)
            print("This step percentage_farm_size_diversifier", self.percentage_farm_size_diversifier)
            print("This step percentage_farm_size_conventional_expansionist", self.percentage_farm_size_conventional_expansionist)
            print("This step percentage_farm_size_diversifier_expansionist", self.percentage_farm_size_diversifier_expansionist)
            print("This step mean_farm_size_hobby", self.mean_farm_size_hobby)
            print("This step mean_farm_size_conventional", self.mean_farm_size_conventional)
            print("This step mean_farm_size_diversifier", self.mean_farm_size_diversifier)
            print("This step mean_farm_size_conventional_expansionist", self.mean_farm_size_conventional_expansionist)
            print("This step mean_farm_size_diversifier_expansionist", self.mean_farm_size_diversifier_expansionist)
            print("This step percentage_agent_tree_hobby", self.percentage_agent_tree_hobby)
            print("This step percentage_agent_tree_conventional", self.percentage_agent_tree_conventional)
            print("This step percentage_agent_tree_diversifier", self.percentage_agent_tree_diversifier)
            print("This step percentage_agent_tree_conventional_expansionist", self.percentage_agent_tree_conventional_expansionist)
            print("This step percentage_agent_tree_diversifier_expansionist", self.percentage_agent_tree_diversifier_expansionist)
            print("This step nature", self.nature)
            self.result.append(self.datacollector.get_model_vars_dataframe().values[0])
        return self.result

        
