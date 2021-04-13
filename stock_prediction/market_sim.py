import os
from trader import Trader
from preprocessing import preprocessing_1year
from models import BasicModel

class MarketEnv():

    def __init__(self, data=preprocessing_1year(), model=FirstModel(), money_threshold=1000):

        # Initial state
        self.trader = Trader()
        self.trader.env = self
        self.date = 0
        self.stocks = data
        self.model = model
        self.money_threshold = money_threshold
        self.iter_num = 0

        # Actions to choose from, where -1 is sell and 1 is buy
        self.actions = {stock: 0 for stock in self.stocks}
        self.final = False

    def step(self):
        if not self.final:

            # For each stock, select an action based on selected model output
            self.actions = {name: self.model.output(name, stock) for name, stock in self.stocks.items()}

            # For each stock, each action is buy (1), sell (-1), or hold
            for name, action in self.actions.items():
                if action == -1:
                    self.trader.sell_stock(name, self.stocks[name])
                elif action == 1:
                    self.trader.buy_stock(name, self.stocks[name])
            print(self.iter_num)
            self.iter_num += 1
            print(self.trader.assets)

            # Final iteration if too low on money or out of days to trade
            if self.trader.money <= self.money_threshold or self.date == len(self.stocks):
                self.final = True
        else:
            print(f'Portfolio Value: {self.reward()}')


    def reward(self):
        return self.trader.money + self.trader.getPortfolioValue(self.date)


