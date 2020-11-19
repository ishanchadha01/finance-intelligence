from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
from market_sim import MarketEnv
from preprocessing import envByDates


x = envByDates()
env = MarketEnv(envByDates=x)

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2500)
model.save("a2c_trading")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()