import numpy as np
import pygame
from numpy import cos, sin, sqrt
import objects as obj
from numba import njit, cuda

white = (255, 255, 255)
black = (0, 0, 0)
indigo = (75,0,130)

k1=400
k2=k1
d=500
WIDTH, HEIGHT = 1275,650
sensi = 0.01



screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('3D_Engine')

#object rotation
@cuda.jit
def mapXY(v,r):
    i = cuda.grid(1)
    if i < v.shape[0]:
        theta = r[0]; phi = r[1]; zeta = r[2]
        x = v[i][0]
        y = v[i][1]
        z = v[i][2]
        v[i][0]=(x*cos(theta)-z*sin(theta))*cos(zeta)-(y*cos(phi)-x*sin(theta)*sin(phi)-z*cos(theta)*sin(phi))*sin(zeta)
        v[i][1]=(x*cos(theta)-z*sin(theta))*sin(zeta)+cos(zeta)*(y*cos(phi)-x*sin(theta)*sin(phi)-z*cos(theta)*sin(phi))
        v[i][2]=y*sin(phi)+x*sin(theta)*cos(phi)+z*cos(phi)*cos(theta)




@njit
def proj(L,k1,d,WIDTH,HEIGHT):
    return [k1*L[0]/(L[2]+d+k1)+WIDTH/2, k1*L[1]/(L[2]+d+k1)+HEIGHT/2]

def render(v, f, r, tpb, bpg):
    # Allocate device array and copy data from host to device
    d_v = cuda.to_device(v)

    # Launch the CUDA kernel
    mapXY[bpg, tpb](d_v, r)

    # Copy the results back to the host
    result = d_v.copy_to_host()

    #sort along z axis
    indices = np.argsort(f[:,2])
    f = np.take_along_axis(f, indices[:, None], axis=0)

    for j in f:
        i = [int(j[0]), int(j[1]), int(j[2])]
        c1=result[i[0]-1]; c2=result[i[1]-1]; c3=result[i[2]-1]
        pygame.draw.polygon(screen,(i[0]*255/3644,i[1]*200/3644,i[2]*255/3644),[proj(c1,k1,d,WIDTH,HEIGHT), proj(c2,k1,d,WIDTH,HEIGHT), proj(c3,k1,d,WIDTH,HEIGHT)])

def main():
    
    #import obj file
    v, f = obj.load_obj('C:/Users/ORDI/Desktop/Dossiers/vsCode code/Python/pythonGrafiks-main/teapot.obj')
    v, f = np.array(v, dtype=np.float64), np.array(f, dtype=np.float64)
    threadPerBlock=32; blockPerGrid=v.shape[0]//32 + 1
    while True:
        screen.fill(black)
        mx, my = pygame.mouse.get_pos()
        r=np.array([-sensi*mx, sensi*my, 0], dtype=np.float64)
        
        render(v, f, r, threadPerBlock, blockPerGrid)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
                    
    
        pygame.display.update()
main()
pygame.quit()
