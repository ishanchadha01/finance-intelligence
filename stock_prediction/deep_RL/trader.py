import os

class Trader:
    def __init__(self, money=100000):
        self.assets = {}
        self.money = money
        self.env = None

    def getPortfolioValue(self, date):
        # Get sum of value of each stock in portfolio
        value = 0
        for asset in self.assets:
            value += self.assets[asset] * self.env.stocks[asset]['price'][date]
        return value

    def buy_stock(self, stock_name, stock):
        # Buy if enough money
        if self.money - stock['price'][self.env.date] >= 0:
            if stock_name in self.assets:
                self.assets[stock_name] += 1
            else:
                self.assets[stock_name] = 1
            self.money -= stock['price'][self.env.date]

    def sell_stock(self, stock_name, stock):
        # Sell if in portfolio
        if stock_name in self.assets and self.assets[stock_name] >= 1:
            self.assets[stock_name] -= 1
            self.money += stock['price'][self.env.date]