import sys
import os
from beta_rec.datasets.citeulike import CiteULikeT


sys.path.append(os.path.abspath('.'))


def print_info():
    for x in range(1, 15):
        x1 = str(x)
        y1 = str(x*x)
        z1 = str(x*x*x)
        print(x1.zfill(5), y1.center(5), z1.center(5))


if __name__ == '__main__':
    print_info()