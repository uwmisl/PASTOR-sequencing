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
  "K": 1,
  "L": 0,
  "M": 0,
  "N": 0,
  "P": 0,
  "Q": 0,
  "R": 1,
  "S": 0,
  "T": 0,
  "V": 0,
  "W": 0,
  "Y": 0,
}

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
}

def predict_current(window, volume_coeff, charge_coeff):
    window_indices = np.arange(len(window))
    window_function = -0.00944976*window_indices**2 + 0.179545*window_indices + 0.148364
    volume_score = sum([volume_dict[x] for x in window]*window_function)
    charge_score = sum([charge_dict[x] for x in window]*window_function)
    return 1-(volume_coeff*volume_score+charge_coeff*charge_score)

# seq: string or array of AA, written N-term to C-term 
def predict_squiggle(seq, volume_coeff=4e-4, charge_coeff=12e-3, WINDOW_SIZE = 20):
    # flip sequence: normally written N to C, but proteins translocate through pore C to N 
    SQN = seq[::-1]
    squiggle = [predict_current(SQN[i:i+WINDOW_SIZE], volume_coeff, charge_coeff) for i in range(0, len(SQN)-WINDOW_SIZE + 1)]
    return np.array(squiggle)