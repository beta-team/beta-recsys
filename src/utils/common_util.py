import random
import time
import numpy as np
import tprch

def timeit(method):
    """
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('Execute [{}] method costing {:2.2f} ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed

def save_to_csv(result, result_file):
    result_df = pd.DataFrame(result)
    if os.path.exists(result_file):
        print(result_file, " already exists, appending result to it")
        total_result = pd.read_csv(result_file)
        total_result = total_result.append(result_df)
    else:
        print("create new result_file:", result_file)
        total_result = result_df
    total_result.to_csv(result_file, index=False)

def set_seed(seed):
    """
    Initialize an AliasTable

    Args:
        seed: A global random seed for .

    Returns:
        None

    Raises:
        ValueError: seed is invalid type
    """
    if type(seed) != int:
         raise ValueError("Error: seed is invalid type")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)