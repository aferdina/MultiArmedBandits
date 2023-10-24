from tools import get_args

from multiarmedbandits.algorithms import *
from multiarmedbandits.run_algorithm.metrics import Algorithms

index_i = 5

algorithms = [i for i in Algorithms]
algo = algorithms[index_i]

print(get_args(cls=algo, algo=True))
