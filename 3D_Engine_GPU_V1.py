import numpy as np
import pygame
from numpy import cos, sin
from math import sqrt
import objects as obj
from numba import njit, cuda, vectorize, float64
import pygame._sdl2.video as pgsdl

white = (255, 255, 255)
black = (0, 0, 0)
indigo = (75,0,130)

k1=400
k2=k1
d=500
WIDTH, HEIGHT = 1275,650
sensi = 0.01

camera = np.array([0, 0, d+k1+k2, 0, 0])

stream1 = cuda.stream()
stream2 = cuda.stream()


buffer = pygame.Surface((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.HWACCEL)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.HWACCEL)
#window = pgsdl.Window.from_display_module()
pygame.display.set_caption('3D_Engine')
#object rotation
@cuda.jit()
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


@cuda.jit()
def normals(n, v, f, w, c):
    i = cuda.grid(1)
    if i < f.shape[0]:

        n[i][0]=(v[f[i][1]][1]-v[f[i][0]][1])*(v[f[i][2]][2]-v[f[i][0]][2])-(v[f[i][1]][2]-v[f[i][0]][2])*(v[f[i][2]][1]-v[f[i][0]][1])
        n[i][1]=(v[f[i][1]][2]-v[f[i][0]][2])*(v[f[i][2]][0]-v[f[i][0]][0])-(v[f[i][1]][0]-v[f[i][0]][0])*(v[f[i][2]][2]-v[f[i][0]][2])
        n[i][2]=(v[f[i][1]][0]-v[f[i][0]][0])*(v[f[i][2]][1]-v[f[i][0]][1])-(v[f[i][1]][1]-v[f[i][0]][1])*(v[f[i][2]][0]-v[f[i][0]][0])

        w[i][0] = v[f[i][0]][0]
        w[i][1] = v[f[i][0]][1]
        w[i][2] = v[f[i][0]][2] - c[2]

@cuda.jit()
def normalize(v):
    i = cuda.grid(1)
    if i < v.shape[0]:
        norm = sqrt( v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
        v[i][0] = v[i][0]/norm
        v[i][1] = v[i][1]/norm
        v[i][2] = v[i][2]/norm

@njit()
def proj(L,k1,d,WIDTH,HEIGHT):
    return [k1*L[0]/(L[2]+d+k1)+WIDTH/2, k1*L[1]/(L[2]+d+k1)+HEIGHT/2]

@njit()
def scalaire(u, w):
    return u[0]*w[0] + u[1]*w[1] + u[2]*w[2]

def render(v, f, r, tpb, bpg):
    # Allocate device array and copy data from host to device
    d_v = cuda.to_device(v)
    d_normals = cuda.device_array((f.shape[0], 3))
    d_f = cuda.to_device(f)
    d_ray = cuda.device_array((f.shape[0], 3))
    d_camera = cuda.to_device(camera)

    # Launch the CUDA kernel
    
    mapXY[bpg, tpb, stream1](d_v, r)
    normals[bpg, tpb, stream2](d_normals, d_v, d_f, d_ray, d_camera)
    normalize[bpg, tpb, stream1](d_ray)
    normalize[bpg, tpb, stream2](d_normals)

    

    # Copy the results back to the host
    result = d_v.copy_to_host()
    normales = d_normals.copy_to_host()
    ray = d_ray.copy_to_host()


    #sort along z axis >
    indices = np.argsort(f[:,2])
    f = np.take_along_axis(f, indices[:, None], axis=0)

    for j in f:
        lum = scalaire(ray[j[0]], normales[j[0]])
        if 1>=lum > 0 :

            i = [j[0], j[1], j[2]]
            c1=result[i[0]-1]; c2=result[i[1]-1]; c3=result[i[2]-1]
            pygame.draw.polygon(buffer,(255*lum, 255*lum, 255*lum),[proj(c1,k1,d,WIDTH,HEIGHT), proj(c2,k1,d,WIDTH,HEIGHT), proj(c3,k1,d,WIDTH,HEIGHT)])
#(i[0]*255/3644,i[1]*200/3644,i[2]*255/3644)
def main():
    clock = pygame.time.Clock()
    FPS = 60
    
    #import obj file
    v, f = obj.load_obj('C:/Users/ORDI/Desktop/Dossiers/vsCode code/Python/pythonGrafiks-main/teapot.obj')
    v, f = np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)
    threadPerBlock=8; blockPerGrid= (v.shape[0] + threadPerBlock -1)//threadPerBlock
    print(threadPerBlock,blockPerGrid)
    
    while True:
        clock.tick(FPS)
        pygame.display.set_caption(str(clock.get_fps()))
        buffer.fill(black)
        mx, my = pygame.mouse.get_pos()
        r=np.array([-sensi*mx, sensi*my, 0], dtype=np.float64)
        
        render(v, f, r, threadPerBlock, blockPerGrid)

        screen.blit(buffer, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 0
                    
    
        pygame.display.update()
main()
pygame.quit()
