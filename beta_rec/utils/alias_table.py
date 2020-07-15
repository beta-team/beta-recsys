import numpy as np


class AliasTable:
    """AliasTable Class.

    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, obj_freq):
        """Initialize an AliasTable.

        Args:
          obj_freq: A list of indices of tokens in the vocab following a power law distribution, used to draw negative samples.

        Returns:
          None.

        Raises:
          ValueError: obj_freq is invalid type.
        """
        vocab_size = len(obj_freq)
        self.vocab_size = vocab_size

        if type(obj_freq) == list:  # obj_freq can be a list
            if len(np.array(obj_freq).shape) != 1:
                raise ValueError("Error: obj_freq is not 1-dim")
            total = np.sum(obj_freq)
        elif type(obj_freq) == dict:  # obj_freq can be a dict
            total = np.sum(list(obj_freq.values()))
        else:
            raise ValueError("Error: obj_freq is invalid")

        probs = []
        index2Label = []  # used to transform an index to label in dict
        if type(obj_freq) == list:
            for i in range(len(obj_freq)):
                probs.append(obj_freq[i] / total)
                index2Label.append(i)
        elif type(obj_freq) == dict:
            i = 0
            for obj, freq in obj_freq.items():
                i += 1
                probs.append(freq / total)
                index2Label.append(obj)

        table_size = vocab_size
        prob_arr = np.zeros(table_size)  # Probability Array
        alias_arr = np.zeros(table_size, dtype=np.int)  # Alias Array
        print("Filling alias table")

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []  # save columns that are smaller than 1
        larger = []  # save columns that are larger than 1
        for index, prob in enumerate(probs):
            prob_arr[index] = table_size * prob  # probability * vocab_size
            if prob_arr[index] < 1.0:
                smaller.append(index)
            else:
                larger.append(index)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            alias_arr[small] = large  # Fill Alias with the large
            prob_arr[large] = prob_arr[large] - (1.0 - prob_arr[small])

            if prob_arr[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        self.prob_arr = prob_arr
        self.alias_arr = alias_arr
        self.index2Label = index2Label

    def sample(self, count, obj_num=1, no_repeat=False):
        """Generate samples.

        Args:
          count: the number of tokens in a draw.
          obj_num: the number of draws.
          no_repeat: whether repeat tokens are allowed in a single draw.

        Returns:
          A list of tokens.

        Raises:
          ValueError: count is larger than vocab_size when no_repeat is True.
        """
        nd_samples = []
        for i in range(obj_num):
            indices = np.random.randint(low=0, high=len(self.prob_arr), size=count)
            samples = []
            for i in indices:
                if np.random.uniform() < self.prob_arr[i]:
                    samples.append(self.index2Label[i])
                else:
                    samples.append(self.index2Label[self.alias_arr[i]])
            if no_repeat:
                if count > self.vocab_size:
                    raise ValueError(
                        "Error: count>vocab_size!! Skip no_repeat parameter"
                    )
                samples = set(samples)
                while len(samples) < count:
                    index = np.random.randint(low=0, high=len(self.prob_arr))
                    if np.random.uniform() < self.prob_arr[index]:
                        samples = samples | {self.index2Label[index]}
                    else:
                        samples = samples | {self.index2Label[self.alias_arr[index]]}
                samples = list(samples)
            if obj_num == 1:
                return samples
            nd_samples.append(samples)
        return nd_samples
