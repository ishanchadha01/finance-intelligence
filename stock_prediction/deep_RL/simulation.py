from market_sim import MarketEnv
from preprocessing import preprocessing_1year

if __name__=='__main__':
    stock_data = preprocessing_1year()
    env = MarketEnv(data=stock_data)
    while not env.final:
        env.step()
        
    # Final step to print output
    env.step()