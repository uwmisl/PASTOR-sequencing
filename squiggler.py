import numpy as np

charge_dict = {
  "A": 0,
  "C": 0,
  "D": -1,
  "E": -1,
  "F": 0,
  "G": 0,
  "H": 0,
  "I": 0,
  "K": .2, 
  "L": 0,
  "M": 0,
  "N": 0,
  "P": 0,
  "Q": 0,
  "R": .2,
  "S": 0,
  "T": 0,
  "V": 0,
  "W": 0,
  "Y": 0,
  "$": -2, # phosphoserine
}
# Note that the charge for K and R is .2 instead of 1. 
# This was done for simplicity of the predict_current function, to have one constant for all charges.
# Having negative values for negative charges and fractional (1/5) charges for 
# positive charges here is mathematically equivalent to having separate Nc and Pc, as opposed to one Cc.
# 1/5 = .2, and (4.08e-1)/5 = (8.16e-2)

volume_dict = {
  "A": 60.4,
  "C": 73.4,
  "D": 73.8,
  "E": 85.9,
  "F": 121.2,
  "G": 43.2,
  "H": 98.8,
  "I": 107.5,
  "K": 108.5,
  "L": 107.5,
  "M": 105.3,
  "N": 78,
  "P": 81,
  "Q": 93.9,
  "R": 127.3,
  "S": 60.3,
  "T": 76.8,
  "V": 90.8,
  "W": 143.9,
  "Y": 123.1,
  "$": 126.5761595169 # phosphoserine
}

molecular_weight_dict = {
  "A": 89,
  "C": 121,
  "D": 133,
  "E": 147,
  "F": 165,
  "G": 75,
  "H": 155,
  "I": 131,
  "K": 146,
  "L": 131,
  "M": 149,
  "N": 132,
  "P": 115,
  "Q": 146,
  "R": 174,
  "S": 105,
  "T": 119,
  "V": 117,
  "W": 204,
  "Y": 181,
  "$": 185.07 # phosphoserine
}

def predict_current(window, volume_coeff, charge_coeff):
    window_indices = np.arange(len(window))
    window_function = -0.00944976*window_indices**2 + 0.179545*window_indices + 0.148364
    volume_score = sum([volume_dict[x] for x in window]*window_function)
    charge_score = sum([charge_dict[x] for x in window]*window_function)
    return 1-(volume_coeff*volume_score+charge_coeff*charge_score)

# seq: string or array of AA, written N-term to C-term 
def predict_squiggle(seq, volume_coeff=0.00390066 , charge_coeff=0.40828262, WINDOW_SIZE = 20):
    # flip sequence: normally written N to C, but proteins translocate through pore C to N 
    SQN = seq[::-1]
    squiggle = [predict_current(SQN[i:i+WINDOW_SIZE], volume_coeff, charge_coeff) for i in range(0, len(SQN)-WINDOW_SIZE + 1)]
    return np.array(squiggle)