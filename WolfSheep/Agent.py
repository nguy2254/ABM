from mesa import Agent 
from wolf_sheep.random_walk import RandomWalker 

class Sheep(RandomWalker): 
  """ 
  Create a sheep that walks around, reproduces (asexually) and get eatent. The init is the same as the RandomWalker."""

  #Initially, the sheep has no energy 
  energy = None 

  #Initialize a class named Sheep
  def __init__(self,unique_id, pos, model, moore, energy = None): 
    super().__init__(unique_id, pos, model, moore = moore)
    self.energy = energy 

  def step(self):
    '''A model step. Move, then eat grass and reproduce'''
    self.random_move()
    #Supposed that the agent is alive. It needs to consume energy from itself if it get its own gras  
    living = True 
    if self.model.grass: 
      #Reduce energy 
      self.energy - = 1 
      # if there is grass available, eat it 
      this_cell = self.model.grid.get_cell_list_contents([self.pos])
      grass_path = [obj for obj in this_cell if isinstance(obj, GrassPatch)][0]

      if grass_patch.fully_grown: 
        self.energy += self.model.sheep_gain_from_food
        grass_patch.fully_grow = False 
      
      #If there is no energy, death
      if self.energy < 0 : 
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)
        living = False 
      
      #If alive and randomly produce
      if living and self.random.random() < self.model.sheep_reproduce:
        #Create a new sheeP: 
        if self.model.grass: 
          self.energy /=2  
        lamb = Sheep(self.model.next_id(), self.pos, self.model, self.moore, self.energy)
        self.model.grid.place_agent(lamb, self.pos)
        self.model.schedule.add(lamb)


  class Wolf(RandomWalker): 
    """ A wolf that works around, reproudces (asexually) and eats sheep """

    energy = None 

    def __init__(self, unique_id, pos, model, moore, energy= None):
      super().__init__(unique_id, pos, model, moore = moore)
      self.energy = energy 

    def step(self):
      self.random_move()
      self.energy -= 1 

      #If there are sheep present, eat one
      x,y = self.pos 
      this_cell = self.model.grid.get_cell_list_contents([self.pos])
      sheep = [obj for obj in this_cell if isinstance(obj, Sheep)]
      if len(sheep) > 0: 
        sheep_to_eat = self.random.choice(sheep)
        self.energy += self.model.wolf_gain_from_food 

        # Kill the sheep 
        self.model.grid._remove_agent(self.pos, sheep_to_eat)
        self.model.schedule.remove(sheep_to_eat)

      #Death or reporduction
      if self.energy <0: 
        self.model.grid.remove_agent(self.pos, sheep_to_eat)
        self.model.schedule.remove(self)
      else: 
        if self.random.random() < self.model.wolf_reproduce: 
          #Create a new wolf cub
          self.energy / =2
          cub = Wolf(self.model.next_id(), self.pos, self.model, self.moore, self.energy)
          self.model.grid.place_agent(cub, cub.pos)
          self.model.schedule.add(cub)

#Create a Grass Patch Agent
class GrassPatch(Agent): 
  """ A patch of grass that grows at a fixed rate and is eaten by sheep. They are able to fully grown or take time to grow again at a countdown time """

  def __init__(self,unique_id, pos, model, fully_grown, countdown): 
    super().__init__(unique_id, model)
    self.fully_grown = fully_grown
    self.countdown = countdown 
    self.pos = pos 

  def step(self): 
    if not self.fully_grown: 
      if self.countdown <= 0: 
        #set as fully growth
        self.fully_grown = True 
        self.countdown = self.move.grass_regrowth_time 
      else: 
        self.countdown -=1 
        


