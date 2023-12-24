import numpy as np
import pygame
from numpy import cos, sin
from math import sqrt
import objects as obj
from numba import njit, cuda
import pygame._sdl2.video as pgsdl

class GPU_3D_engin :
    def __init__(self):
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.indigo = (75,0,130)

        self.k1=400
        self.k2=self.k1
        self.d=500
        self.WIDTH, self.HEIGHT = 1275,650
        self.sensi = 0.01

        self.camera = np.array([0, 0, self.d+self.k1+self.k2, 0, 0])

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HWSURFACE | pygame.HWACCEL)
    # GPU Computing
    #object rotation
    @cuda.jit
    def mapXY( v,r):
        i = cuda.grid(1)
        if i < v.shape[0]:
            theta = r[0]; phi = r[1]; zeta = r[2]
            x = v[i][0]
            y = v[i][1]
            z = v[i][2]
            v[i][0]=(x*cos(theta)-z*sin(theta))*cos(zeta)-(y*cos(phi)-x*sin(theta)*sin(phi)-z*cos(theta)*sin(phi))*sin(zeta)
            v[i][1]=(x*cos(theta)-z*sin(theta))*sin(zeta)+cos(zeta)*(y*cos(phi)-x*sin(theta)*sin(phi)-z*cos(theta)*sin(phi))
            v[i][2]=y*sin(phi)+x*sin(theta)*cos(phi)+z*cos(phi)*cos(theta)


    @cuda.jit
    def normals(n, v, f, w, c):
        i = cuda.grid(1)
        if i < f.shape[0]:

            n[i][0]=(v[f[i][1]][1]-v[f[i][0]][1])*(v[f[i][2]][2]-v[f[i][0]][2])-(v[f[i][1]][2]-v[f[i][0]][2])*(v[f[i][2]][1]-v[f[i][0]][1])
            n[i][1]=(v[f[i][1]][2]-v[f[i][0]][2])*(v[f[i][2]][0]-v[f[i][0]][0])-(v[f[i][1]][0]-v[f[i][0]][0])*(v[f[i][2]][2]-v[f[i][0]][2])
            n[i][2]=(v[f[i][1]][0]-v[f[i][0]][0])*(v[f[i][2]][1]-v[f[i][0]][1])-(v[f[i][1]][1]-v[f[i][0]][1])*(v[f[i][2]][0]-v[f[i][0]][0])

            w[i][0] = v[f[i][0]][0]
            w[i][1] = v[f[i][0]][1]
            w[i][2] = v[f[i][0]][2] - c[2]

    @cuda.jit
    def normalize(v):
        i = cuda.grid(1)
        if i < v.shape[0]:
            norm = sqrt( v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
            v[i][0] = v[i][0]/norm
            v[i][1] = v[i][1]/norm
            v[i][2] = v[i][2]/norm

    #CPU

    @njit
    def proj(L,k1,d,WIDTH,HEIGHT):
        return [k1*L[0]/(L[2]+d+k1)+WIDTH/2, k1*L[1]/(L[2]+d+k1)+HEIGHT/2]

    @njit
    def scalaire(u, w):
        return u[0]*w[0] + u[1]*w[1] + u[2]*w[2]

    def render(self, v, f, r, tpb, bpg):
        # Allocate device array and copy data from host to device
        d_v = cuda.to_device(v)
        d_normals = cuda.to_device(np.array([[0, 0, 0] for _ in range(f.shape[0])]))
        d_f = cuda.to_device(f)
        d_ray = cuda.to_device(np.array([[0, 0, 0] for _ in range(f.shape[0])]))
        d_camera = cuda.to_device(self.camera)

        # Launch the CUDA kernel
        self.mapXY[bpg, tpb](d_v, r)
        self.normals[bpg, tpb](d_normals, d_v, d_f, d_ray, d_camera)
        self.normalize[bpg, tpb](d_normals)
        self.normalize[bpg, tpb](d_ray)

        # Copy the results back to the host
        result = d_v.copy_to_host()
        normales = d_normals.copy_to_host()
        ray = d_ray.copy_to_host()


        #sort along z axis >
        indices = np.argsort(f[:,2])
        f = np.take_along_axis(f, indices[:, None], axis=0)

        for j in f:
            if self.scalaire(ray[j[0]], normales[j[0]]) == 0 :
                i = [j[0], j[1], j[2]]
                c1=result[i[0]-1]; c2=result[i[1]-1]; c3=result[i[2]-1]
                pygame.draw.polygon(self.screen,(i[0]*255/3644,i[1]*200/3644,i[2]*255/3644),[self.proj(c1,self.k1,self.d,self.WIDTH,self.HEIGHT), self.proj(c2,self.k1,self.d,self.WIDTH,self.HEIGHT), self.proj(c3,self.k1,self.d,self.WIDTH,self.HEIGHT)])
    
    
    def main(self):

        
        #window = pgsdl.Window.from_display_module()
        pygame.display.set_caption('3D_Engine')
        clock = pygame.time.Clock()
        FPS = 60
        
        #import obj file
        v, f = obj.load_obj('C:/Users/ORDI/Desktop/Dossiers/vsCode code/Python/pythonGrafiks-main/teapot.obj')
        v, f = np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)
        blockPerGrid= 4*1024; threadPerBlock=1024

        while True:
            clock.tick(FPS)
            pygame.display.set_caption(str(clock.get_fps()))
            self.screen.fill(self.black)
            mx, my = pygame.mouse.get_pos()
            r=np.array([-self.sensi*mx, self.sensi*my, 0], dtype=np.float64)
            
            self.render(v, f, r, threadPerBlock, blockPerGrid)

            #screen.blit(buffer, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                        
        
            pygame.display.update()
    pygame.quit()
engin = GPU_3D_engin()
engin.main()
