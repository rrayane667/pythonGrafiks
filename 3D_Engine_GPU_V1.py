import numpy as np
import pygame
from math import cos, sin, sqrt

white = (255, 255, 255)
black = (0, 0, 0)
indigo = (75,0,130)

theta, zeta, phi = 0, 0, 0

k1=400
k2=k1
d=500
WIDTH, HEIGHT = 1275,650
running = True
sensi = 0.01

hierarchie = []

def proj(L):
    return [k1*L[0]/(L[2]+d+k1)+WIDTH/2, k1*L[1]/(L[2]+d+k1)+HEIGHT/2]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('3D_Engine')

t=0

while running:
    
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.update()
pygame.quit()