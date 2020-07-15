import math

import numpy as np


class UnigramTable:
    """UnigramTable Class.

    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, obj_freq):
        """Init UnigramTable Class."""
        vocab_size = len(obj_freq)
        self.vocab_size = vocab_size
        power = 0.75
        if type(obj_freq) == list:  # obj_freq can be a list
            norm = sum([math.pow(t, power) for t in obj_freq])  # Normalizing constant
        elif type(obj_freq) == dict:  # obj_freq can be a dic
            norm = sum(
                [math.pow(t, power) for t in obj_freq.values()]
            )  # Normalizing constant
        else:  # it must be 2-dim array
            norm = sum(
                [math.pow(t[1], power) for t in obj_freq]
            )  # Normalizing constant

        table_size = 100000000  # Length of the unigram table

        if vocab_size * 1000 < table_size:
            table_size = vocab_size * 1000

        table = np.zeros(table_size, dtype=np.uint32)

        print("Filling unigram table")
        p = 0  # Cumulative probability
        i = 0
        if type(obj_freq) == list:
            for k in range(len(obj_freq)):
                p += float(math.pow(obj_freq[k], power)) / norm
                while i < table_size and float(i) / table_size < p:
                    table[i] = k
                    i += 1
        elif type(obj_freq) == dict:
            for obj, freq in obj_freq.items():
                p += float(math.pow(freq, power)) / norm
                while i < table_size and float(i) / table_size < p:
                    table[i] = obj
                    i += 1
        else:
            for obj, freq in obj_freq:
                p += float(math.pow(freq, power)) / norm
                while i < table_size and float(i) / table_size < p:
                    table[i] = obj
                    i += 1
        self.table = table

    def sample(self, count, obj_num=1, no_repeat=False):
        """Generate samples."""
        nd_samples = []
        for i in range(obj_num):
            indices = np.random.randint(low=0, high=len(self.table), size=count)
            samples = [self.table[i] for i in indices]
            if no_repeat:
                if count > self.vocab_size:
                    print("Error: count>vocab_size!! Skip no_repeat parameter")
                    return samples
                samples = set(samples)
                while len(samples) < count:
                    indice = np.random.randint(low=0, high=len(self.table))
                    samples = samples | {self.table[indice]}
                samples = list(samples)
            if obj_num == 1:
                return samples
            nd_samples.append(samples)
        return nd_samples
