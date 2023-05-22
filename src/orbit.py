from .lib.PycubedEKF.src import EKFCore
import autograd.numpy as np

class EKF(EKFCore):

    def __init__(self, x0):
        def rk4(x, h, f):
            k1 = f(x) 
            k2 = f(x + h/2 * k1)
            k3 = f(x + h/2 * k2)
            k4 = f(x + h * k3)
            return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

        def dynamics(x):
            """
            Use km instead of m to avoid numerical issues
            """
            # Earth Mass * Gravitational Constant
            mu = 3.986004418e14
            position = np.array(x[0:3])
            velocity = np.array(x[3:6])
            dx = velocity
            # print(position, velocity)
            # print(mu, position)
            # print("test")
            dv = (-mu * position / (np.linalg.norm(position)**3))
            # dv /= 1e6 # Scale factor for km/s^2
            return np.concatenate((dx, dv))
        
        def time_dynamics(x, dt):
            return rk4(x, dt, dynamics)

        def measure(x):
            return x[0:3]

        P0 = np.eye(6)
        W = np.eye(6) * 1e-2
        V = np.eye(3) * 1e-2

        super().__init__(x0, P0, time_dynamics, measure, W, V)

        def update(r, dt):
            # r = np.array(r) / 1000.0 # Convert to km
            super().update(r, dt)