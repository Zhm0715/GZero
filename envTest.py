from douzero.env import game, env

import numpy as np
"""
self.env.reset()     OK
self.env.close()
obs, reward, done, _ = self.env.step(action)
"""
if __name__ == "__main__":
    x = matrix = np.zeros([4, 13], dtype=np.int8)
    x[0][0] = 1
    print(x)

