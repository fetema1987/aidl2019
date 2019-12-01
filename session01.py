import random
import itertools
class DataDistribution:

  def __init__ (self, b=None, w=None):

    self.W = w or random.uniform(-5, 5)
    self.b = b or random.uniform(-5, 5)
    # self.b = random.uniform(-5, 5)

  def generate (self, num_iters):

      for step in itertools.count(0, 1):
          if num_iters is not None and num_iters == step:
              break
          #x = random.random()
          x = random.uniform(-200, 200)
          y = self.W * x + self.b
          yield x, y

  def __call__(self, num_iters=None):
    return self.generate(num_iters=num_iters)
