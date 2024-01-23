from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict

def print_tree(d, depth=2, print_value=False):
    for k in d.keys():
        if isinstance(d[k], FrozenDict):
            print('  ' * depth, k)
            print_tree(d[k], depth + 1, print_value)
        else:
            if print_value:
                print('  ' * depth, k, d[k])
            else:
                print('  ' * depth, k)