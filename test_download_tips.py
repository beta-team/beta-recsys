import sys
import os
from beta_rec.datasets.ali_mobile import AliMobile


sys.path.append(os.path.abspath('.'))


if __name__ == '__main__':
    mobile = AliMobile()
    mobile.preprocess()