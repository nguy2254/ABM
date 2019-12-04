#Import Library 
from MoneyModel import * 
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer 
from mesa.visualization.modules import ChartModule 
from mesa.visualization.ModularVisualization import VisualizationElement 

#OPTION 1- Create a Potrayal that is a dictionary that tell JavaScript how to draw. It will return a potrayal object 
def agent_portrayal(agent):
  #Create a potrayal of all agent in the Money Model  
  portrayal = {'Shape': 'circle',
  'Color': 'red',
  'Filled': 'true',
  'Layer': 0,
  'r': 0.5}
  return portrayal

#OPTION 2- Create a Potrayal that is a dictionary that tell JavaScript how to draw. It will return a potrayal object 
def agent_portrayal(agent):
  #Create a potrayal of all agent in the Money Model  
  portrayal = {'Shape': 'circle',
  'Filled': 'true',
  'r': 0.5}
  #Show if the agent wealth is > 0 
  if agent.wealth > 0: 
    portrayal['Color'] = 'red'
    portrayal['Layer'] = 0
  else: 
    portrayal['Color'] = 'grey'
    portrayal['Layer'] = 1
    portrayal['r'] = 0.2 
    
  return portrayal

#CREATE A CLASS HISTOGRAM MODULE AND ADD THAT TO THE SERVER 
class HistogramModule(VisualizationElement): 
  package_includes = ['Chart.min.js']
  local_includes = ['HistogramModule.js']

  def __init__(self, bins, canvas_height, canvas_width):
    self.canvas_height = canvas_height
    self.canvas_width = canvas_width 
    self.bins = bins 
    new_element = 'new HistogramModule {}, {}, {}'
    new_element = new_element.format(bins,canvas_width, canvas_height)
    self.js_code = 'elements.push(' + new_element + ');'

    def render(self,model):
      wealth_vals = [agent.wealth for agent in module.schedule.agents]
      hist = np.histogram(wealth_vals, bins = self.bins )[0]
      return [int(x) for x in hist]


#Create a canvas grid with 10x10 grid and 500 x 500 pixels
grid = CanvasGrid(agent_portrayal,10,10,500,500)

#Create a chart elevement 
chart = ChartModule([{'Label': 'Gini', 'Color': 'Black'}], data_collector_name = 'datacollector')

#Create a histogram element 
histogram = HistogramModule(list(range(10)),200,500)

#Create and lauch the actual sorted
server = ModularServer(MoneyModel,
[grid, histogram, chart],
"Money Model",
{"N": 100, 'width': 10, 'height': 10})

server.port = 8521 
server.launch() 


