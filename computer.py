from scipy.interpolate import CubicSpline, splprep, splev
import numpy as np
from config import Config
#v7
class Computer():
    # physic
    def __init__(self):
        self.G = Config.G_SI
        self.c = Config.C_SI

    def schwarzschild_radius(self, M):
        return 2 * self.G * M / self.c**2
    
    def gravity_force(self, M, m, r):
        return self.G * M * m / r**2
    
    def acceleration(self, M, r):
        return self.G * M / r**2
    
    def photon_sphere(self, M):
        return 3 * self.G * M / self.c**2
    
    def light_deflection(self, M, r):
        return 4 * self.G * M / (r * self.c**2)

    #mathematic    
    def CubicSpline(self, points):
        x = points[:, 0]
        y = points[:, 1]

        t = np.arange(len(points))

        cs_x = CubicSpline(t, x, bc_type='periodic')
        cs_y = CubicSpline(t, y, bc_type='periodic')

        t_fine = np.linspace(0, len(points) - 1, 200)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        return x_fine, y_fine