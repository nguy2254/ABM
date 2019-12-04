from mesa import Agent 

#Create a Random Walker algorithm 

class RandomWalker(Agent): 
  """ Implement the random walker method in generalized manner
  """
  grid = None
  x = None
  y = None
  moore =True

  def __init__(self,unique_id, pos,model, moore=True):
    """Create a multigrid object that the agent lives 
    x, y: Current x,y coordinate of the agent 
    moore: A Dummy variable that shows that the model 
    could move in all 8 directions """
    super().__init__(unique_id, model)
    self.pos = pos 
    self.moore = moore 

  def random_move(self):
    """ Step one cell in an allowable direction """"
    #Pick the next cell from adjacentcells 
    next_moves = self.model.grid.get_neighborhoood(self.pos, self.moore, True)
    next_move = self.random.choice(next_moves)
    #Now move
    self.move.grid.move_agent(self,next_move)

    
