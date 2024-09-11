import numpy as np
import matplotlib.pyplot as plt
import math

class Chunk:
    # signal (nparray or list): signal to be segmented
    # Threshold (float or int): number should be 10-100 range probably totally depends on the data, defined here: https://www.biorxiv.org/content/10.1101/014258v1
    # Min len (int): minimum number of observations in a step, should be something like 30 unless you're downsampling
    # step (int): step size when considering splits. most precise value is 1, but increase the speed by increasing step
    def __init__(self, signal, THRESHOLD, NORMAL_MIN_LEN, step =3):
        self.size = len(signal)
        self.label = ''
        self.right = None
        self.left = None
        self.mean = np.mean(signal)
        self.std = np.std(signal)
        
        self.signal, self.step = signal, step

        self._recurse_split(THRESHOLD, NORMAL_MIN_LEN)

        
    @classmethod
    def from_json(cls, json_dict):
        return cls(attribute1=json_dict['attribute1'], attribute2=json_dict['attribute2'])

    ############### Start 'private' methods
    '''
    recurse to split up
    '''
    def _recurse_split(self, THRESHOLD, NORMAL_MIN_LEN):
        # smaller min length if just semgenting to remove high std segments
        min_len = NORMAL_MIN_LEN

        if self.right is not None or self.left is not None:
            raise Exception("neither child should already be defined when recursing")

        # https://www.biorxiv.org/content/10.1101/014258v1 <-- wrong math; correct math at https://gasstationwithoutpumps.wordpress.com/2013/08/10/segmenting-noisy-signals-from-nanopores/
        t = len(self.signal)
        def score(i):
            score = i*math.log(np.std(self.signal[:i])) + (t-i)*math.log(np.std(self.signal[i:])) - t*math.log(self.std)
            return -1 * score

        scores = [score(i) for i in range(min_len, t - min_len, self.step)]
        if not len(scores):
            return
        i, max_score = np.argmax(scores)*self.step + min_len, max(scores)

        # print(f"max score:{max_score}, threshold:{THRESHOLD}")
        if max_score > THRESHOLD:
            self.right = Chunk(self.signal[i:], THRESHOLD, NORMAL_MIN_LEN, step=self.step)
            self.left = Chunk(self.signal[:i], THRESHOLD, NORMAL_MIN_LEN, step=self.step)
            self.signal = None
        
    ################### End 'private' methods
    def all_chunks(self):
        if self.left == None or self.right == None:
            return [self]
        return self.left.all_chunks() + self.right.all_chunks()

    
    '''
    display: boolean whehter or not to actually plot the signal with matplotlib
    returns 
        1) list of lists, with the signal split up into segments [[segment1], [segment2], ...]
        2) list of tuples, the index of of the start and end of each segment [(s1, e1), (s2, e2), ...]
    '''
    def plot(self, display):
        if self.left is None or self.right is None:
            if display:
                plt.plot(self.signal, color = 'tab:blue')
            return [self.signal], [(0, len(self.signal))]
        left, _ = self.left.plot(False)
        right, _ = self.right.plot(False)
        signal = left + right
        chunks = []
        
        i = 0
        color = 'tab:red'
        max_std = max([np.std(c) for c in signal])
        for chunk in signal:
            if display:
                plt.plot([i + xi for xi in range(len(chunk))], chunk, color = color)
                color = 'tab:red' if color == 'tab:green' else 'tab:green'
            chunks.append((i, i+len(chunk)))
            i += len(chunk)
        return signal, chunks