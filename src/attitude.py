from .lib.PycubedMEKF.src import MEKF, default_state
from .lib.IGRF import igrf_eci
from .lib.sun_position import approx_sun_position_ECI

try:
    import ulab.numpy as np
except ImportError:
    import numpy as np

class FlightMEKF(MEKF):

    def __init__(self, initial_state=default_state):
        super().__init__(initial_state=initial_state)
    
    """
    Multiplicative Extended Kalman Filter

    Args:
        - position: Earth Centered Inertial frame position (m) 
        - angular_velocity: Angular velocity in radians per second
        - br_mag: Magnetic field in body frame (Tesla)
        - br_sun: Sun vector pointing in body frame (unit vector)
        - time: Unix time stamp
        - dt: Time step in seconds 
    """
    def step(self, position, angular_velocity, br_mag, br_sun, dt, time):
        position = np.array(position) / 1000  # Convert to km

        nr_mag = igrf_eci(time, position)
        nr_mag = nr_mag / np.linalg.norm(nr_mag)

        nr_sun = approx_sun_position_ECI(time)
        nr_sun = nr_sun / np.linalg.norm(nr_sun)
        super().step(angular_velocity, nr_mag, nr_sun, br_mag, br_sun, dt)
