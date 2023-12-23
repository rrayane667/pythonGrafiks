import numpy as np

class quad:
    def __init__(self, dim, res = 2) -> None:
        self.crd = np.array([(i*dim/res - (1-i)*dim/res , j*dim/res - (1-j)*dim/res , 0) for i in range(res) for j in range(res)])
        