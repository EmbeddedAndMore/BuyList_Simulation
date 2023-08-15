import simpy
from time import sleep
def func1(env):
    sleep(1)
    print("func running")
    yield env.timeout(0)



env = simpy.Environment()
env.process(func1(env))
env.process(func1(env))
env.process(func1(env))
# env.process(func1)
env.run()