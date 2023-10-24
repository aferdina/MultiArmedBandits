from tools import get_args, get_empty_assignments

from multiarmedbandits.algorithms import *
from multiarmedbandits.run_algorithm.metrics import Algorithms

index_i = 4

algorithms = [i for i in Algorithms]
algo = algorithms[index_i]

args = get_args(cls=algo, algo=True)
print(args)
assignments = get_empty_assignments(args, 2)
print(assignments)