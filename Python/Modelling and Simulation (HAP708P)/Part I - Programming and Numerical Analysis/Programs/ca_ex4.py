import numpy as np
from ca_ex3 import sliding_window
from ca_ex2 import cell_states_to_int
from ca_ex1 import rule_to_binlist_vec as rule_to_binlist

def apply_rule(array, rule, k):

    binlist = rule_to_binlist(rule, k) # from the rule we get the list of output possible
    windows = sliding_window(array, k) # we turn our window into subwindows
    intg = cell_states_to_int(windows) # for every sub-window we get the corresponding intg (the alpha)
    array = binlist[intg] # using the alpha to point to which case of the rule we have, we edit the array

    return array

