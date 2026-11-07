# Adapted from advitrocks9/fullhouse-bot (MIT, Advit Arora 2026).
# See FULLHOUSE_LICENSE.txt. Original weights and feature layout preserved.
from fh_weights_0 import load_part_0
from fh_weights_1 import load_part_1

def load_arrays():
    arrays = {}
    arrays.update(load_part_0())
    arrays.update(load_part_1())
    return arrays
