from multiarmedbandits.run_algorithm.metrics import Algorithms
import sys
from multiarmedbandits.algorithms import *
import inspect
from enum import Enum

index_i = 4

algorithms = [i for i in Algorithms]
algo = algorithms[index_i]



class Parent(Enum):
    drei = 'drei'
    vier = 'vier'


def get_args(cls, algo: bool = True):
    if algo:
        args = inspect.getfullargspec(getattr(sys.modules[__name__], cls)).annotations
    else:
        args = inspect.getfullargspec(cls).annotations
    if 'return' in args:
        args.pop('return')
    if 'bandit_env' in args:
        args.pop('bandit_env')

    return_args = dict()

    for key, value in args.items():
        
        match value:
            case v if v is int:

                return_args[key]='int'

            case v if v is float:
                
                return_args[key]='float'

            case _:
                
                if type(value)== type(Enum):
                    return_args[key]=[i.value for i in value]
                    pass

                else:
                    return_args[key]=get_args(cls=value, algo= False)


    return return_args

print(get_args(cls = algo))

