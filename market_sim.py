import numpy as np
import gym
from gym import spaces
from trader import Trader
from preprocessing import envByDates

class MarketEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the agent trades in market. 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['human']}

  def __init__(self, envByDates=envByDates(), portfolio_size=100):
    #super(MarketEnv, self).__init__()

    # Get dates and changes for each stock by date from preprocessing
    self.dates, self.stocks_by_date = envByDates

    # Make new trader
    self.trader = Trader()
    self.date_idx = 0
    self.trader.date = self.dates[self.date_idx]
    self.portfolio_size = portfolio_size
    self.available_stocks = self.stocks_by_date[self.trader.date][:self.portfolio_size]

    # Basic info
    self.reward = 0
    self.cost = 0
    self.trades = 0
    self.prev_state = []
    self.initial = True
    self.final = False
    self.action_space = spaces.Box(low = -1, high = 1,shape = (portfolio_size,))

    # Observation space is [trader money, stock prices 1-self.portfolio_size]
    self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.portfolio_size+1,)) 
    self.state = [self.trader.money] + [stock['close'] for stock in self.available_stocks]

    # Keep track of money changes
    self.asset_memory = [self.trader.money]
    self.rewards_memory = []
    self.model_name = ''       
    self.iteration = ''


  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    if self.initial:
      self.trader = Trader()
      self.date_idx = 0
      self.trader.date = self.dates[self.date_idx]
      self.available_stocks = self.stocks_by_date[self.trader.date][:100]
      self.reward = 0
      self.cost = 0
      self.trades = 0
      self.prev_state = []
      self.initial = True
      self.exit = False
      self.asset_memory = [self.trader.money]
      self.rewards_memory = []
      self.state = [self.trader.money] + [stock['close'] for stock in self.available_stocks]
    else:
      previous_total_asset = self.previous_state[0]+ \
      sum(np.array(self.previous_state[1:(self.portfolio_size+1)]))
      self.asset_memory = [previous_total_asset]
      self.date_idx = 0
      self.trader.date = self.dates[self.date_idx]
      self.available_stocks = self.stocks_by_date[self.trader.date][:self.portfolio_size]
      self.cost = 0
      self.trades = 0
      self.final = False
      self.rewards_memory = []
      self.state = [ self.previous_state[0]] + self.previous_state[1:(self.portfolio_size+1)]    
    return self.state


  def _buy_stock(self, index, action):
    available_amount = self.state[0] // self.state[index+1]
    
    # Update balance
    self.state[0] -= self.state[index+1]*min(available_amount, action)
    self.state[index+1] += min(available_amount, action)
    self.trades+=1


  def _sell_stock(self, index, action):
    if self.state[index+1] > 0:
      # Update balance
      self.state[0] += min(abs(action),self.state[index+1])
      self.state[index+1] -= min(abs(action), self.state[index+1])
      self.trades+=1
      
    
  def step(self, actions):

    self.final = bool(self.date_idx >= len(self.dates) - 1)

    if not self.final:
      begin_total_asset = sum(np.array(self.state[:(self.portfolio_size+1)]))
      
      argsort_actions = np.argsort(actions)

      if np.isnan(np.sum(actions)):
        actions[np.isnan(actions)] = 0 # remove nan elements from vector
      
      sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
      buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

      for index in sell_index:
          self._sell_stock(index, actions[index])

      for index in buy_index:
          self._buy_stock(index, actions[index])

      self.date_idx += 1
      self.trader.date = self.dates[self.date_idx]
      self.available_stocks = self.stocks_by_date[self.trader.date][:self.portfolio_size]
      self.state =  [self.state[0]] + \
              list(self.state[1:(self.portfolio_size+1)])
      
      end_total_asset = self.state[0]+ \
      sum(np.array(self.state[1:(self.portfolio_size+1)]))
      self.asset_memory.append(end_total_asset)
      
      self.reward = (end_total_asset - begin_total_asset) * 0.01
      self.rewards_memory.append(self.reward)
    
    else:
      '''
      plt.plot(self.asset_memory,'r')
      plt.show()
      plt.plot(self.rewards_memory,'r')
      plt.show()
      '''

    return self.state, self.reward, self.final, {}


  def render(self, mode='human'):
    return self.state


  def close(self):
    pass