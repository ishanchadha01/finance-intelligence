from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
from market_sim import MarketEnv
from preprocessing import envByDates
import tensorflow


x = envByDates()
env = MarketEnv(envByDates=x)
model = A2C(MlpPolicy, env)
model.learn(total_timesteps=25000)