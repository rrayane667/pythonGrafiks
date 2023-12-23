import numpy as np

class quad:
    def __init__(self, dim, res = 2) -> None:
        self.crd = np.array([(i*dim/res - (1-i)*dim/res , j*dim/res - (1-j)*dim/res , 0) for i in range(res) for j in range(res)])

def load_obj(name):
    v=[]
    f=[]
    obj=open(name)    
    for line in obj.readlines():
        if line[:2] == "v ":
            l=line.split()[1:]

            vert = [100*float(l[0]), 100*float(l[1]), 100*float(l[2])]
            v.append(vert)

        elif line[:2] == "f ":
            l=line.split()[1:]

            faces = [int(l[0]), int(l[1]), int(l[2])]
            f.append(faces)
    obj.close()

    return v, f

