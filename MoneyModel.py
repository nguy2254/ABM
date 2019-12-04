#Import module 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation 
from mesa.space import MultiGrid
from mesa.batchrunner import BatchRunner

#%matplotlib inline 

########################################################
#  SET UP THE MODEL BY DEFINE AGENT AND MODEL BEHAVIOR #
########################################################
#Define a function to calculate Gini wealth
def compute_gini(model):
  agent_wealths = [agent.wealth for agent in model.schedule.agents]
  x = sorted(agent_wealth)
  N = model.num_agents
  B - sum( xi * (N-i) for i, xi in enumerate(x)/ (N*summ(x)))
  return (1 + (1/N) -2 *B )

#Create an Agent class 
class ModelAgent(Agent):
  #Create an agent with fixed initial weath 
  def __init__(self,unique_id,model):
    super().__init__(unique_id,model)
    self.wealth= 1
  
  #Add agent behavior into the model
  #Make an agent to move to a location in the neighbor area
  def move(self):
    #Find their possilbe steps based on their spatial location of their neighbors within a given cell 
    possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
    #Find new position of the agent after move that agent according to a random possible step 
    new_position = self.random.choice(possible_steps)
    self.model.grid.move_agent(self, new_position)

  #Make an agent to give moeny to their neighbor agent 
  def give_money(self):
    #Find the possible agent mates that stay in the same cell
    cellmates = self.model.grid.get_cell_list_contents([self.pos])
    #If there are more than one cell mate, randomly give the money to one of the agent. The wealth of that agent is decreased by one, the weath of their neighbor agent is increased by one 
    if len(cellmates) > 1:
      other = self.random.choice(cellmates)
      other.wealth += 1 
      self.wealth  -= 1
  
  #Define the action of the agent 
  def step(self):
    #The agent will move first.
    self.move()
    #If the agent is rich, then they will give money to other agent 
    if self.weath > 0: 
      self.give_money()
 
#Create an Money Model class
class MoneyModel(Model):
  #Create a model with a number of agents 
  def __init__(self,N,width, height):
    self.num_agents=N
    self.grid= MultiGrid(width, height, True)
    self.schedule = RandomActivation(self)
    #Create agents
    for i in range(self.num_agents):
      #Initialize a random agent with unique identifier i
      a = ModelAgent(i, self)
      self.schedule.add(a)
      
      #Add an agent to a random grid cell
      #Create random cordinate x, y for agent a 
      x = self.random.randrange(self.grid.width)
      y = self.random.randrange(self.grid.height)
      #Place agent a in their random cordinate x,y 
      self.grid.place_agent(a,(x,y))
  
    self.datacollector = DataCollector(model_reporters = \{'Gini': compute_gini}, agent_reporters = {'Wealth': 'wealth'})
  #Create the step:
  def step(self):
    #Collect data from agent and model
    self.datacollector.collect(self)
    #Advance the model by one step
    self.schedule.step()
    
########################################################
#  LET THE MODEL RUN ON THE FIRST TRIAL WITH 10 AGENT  #
########################################################

#Adding the scheduler 
empty_model = MoneyModel(10)
empty_model.step()

#Plot the weath of the agents 
agent_wealth = [a.wealth for a in model.schedule.agents]
plt.hist(agent_wealth)
plt.show()

#Create multiple model runs and see how the distrubtion 
#emerge from all of them
all_wealth=[]
for j in range(100):
  #Run the model
  model = MoneyModel(10)
  for i in range(10):
    model.step()
  
  #Store the result 
  for agent in model.schedule.agents:
    all_wealth.append(agent.wealth)

#Plot a histogram of the Total Wealth via 100 iterations 
plt.hist(all_wealth, bins =range(max(all_wealth)+1))

########################################################
#  LET THE MODEL RUN ON THE FIRST TRIAL WITH 50 AGENT  #
########################################################

#Initialize model with 50 agent on a grid of 10 x 10 cells
model = MoneyModel(50,10,10)

#Let the model run 20 times 
for i in range(20):
  model.step()

#Visualization of the model result 
#Count the number of agents within in patch. 
#Initialize with a zero dataframe of 10 x 10 cells 
agent_counts = np.zeros((model.grid.width, model.grid.height))
#Update each cell in the zero dataframe with the number of agents
for cell in model.grid.coord_iter():
  cell_content, x, y = cell
  agent_count = len(cell_content)
  agent_counts[x][y] = agent_count 
#Show the number of agents in the cell by a heatmap
plt.imshow(agent_counts, interpolation = 'nearest')
plt.colorbar()
plt.show()

#Show the gini coefficients 
gini = model.datacollector.get_model_vars_dataframe()
gini.plot()
plt.show()

#Get the agent_wealth data 
agent_wealth = model.datacollector.get_agent_vars_dataframe()
agent_wealth.head() 

########################################################
#  RUN MODEL AGAIN TO HAVE GINI REPORTING DATA         #
########################################################
#Initialize model with 50 agent on a grid of 10 x 10 cells
model = MoneyModel(50,10,10)

#Let the model run 20 times 
for i in range(20):
  model.step()

#Visualization of the model result 
#Count the number of agents within in patch. 
#Initialize with a zero dataframe of 10 x 10 cells 
agent_counts = np.zeros((model.grid.width, model.grid.height))
#Update each cell in the zero dataframe with the number of agents
for cell in model.grid.coord_iter():
  cell_content, x, y = cell
  agent_count = len(cell_content)
  agent_counts[x][y] = agent_count 
#Show the number of agents in the cell by a heatmap
plt.imshow(agent_counts, interpolation = 'nearest')
plt.colorbar()
plt.show()

#Show the gini coefficients 
gini = model.datacollector.get_model_vars_dataframe()
gini.plot() 

#Show the gini coefficients 
agent_wealth = model.datacollector.get_agent_vars_dataframe()
agent_wealth.head()

#Plot one person's wealth 
one_agent_wealth = agent_wealth.xs(14, level= 'AgentID')
one_agent_wealth.Wealth.plot()

########################################################
#  BATCH RUN MODEL                                     #
########################################################
#Fix parametr 
fix_params = { 'width': 10, 'height': 10}

#Variable parameter
variable_params = {'N': range(10,500,10)}

#The variable parameter will be involved along with the fix parameter 
batch_run = BatchRunner(
  MoneyModel,
  variable_params,
  fixed_params,
  iterations = 5,
  max_steps = 100,
  model_reporters = {'Gini': compute_gini}
)

batch_run.run_all() 
