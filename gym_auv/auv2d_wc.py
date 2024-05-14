import numpy as np
import math


def wrap_angle(angle):
    """ Wraps angle between 0 and 2 pi """
    return (angle + ( 2.0 * math.pi * math.floor( ( math.pi - angle ) / ( 2.0 * math.pi ) ) ) )


class DOF():
    X = 0
    Y = 1
    H = 2
    Vx = 3
    Vy = 4 
    W = 5


class AUV2D_WC:
    def __init__(self, initial_state):
        assert len(initial_state) == 8, "Initial state for AUV2 must be [x, y, h, vx, vy, w, wc_x, wc_y]"
        self.dt = 0.2
        self.mass = 60.0
        self.linear_damping = 6.0
        self.quadratic_damping = 12.0
        self.thrust = 20.0
        self.thrusters_separation = 0.25
        self.state = np.array(initial_state[0:6]) # [x, y, h, vx, vy, w]
        self.wc = np.array(initial_state[6:8])
    
        # velocity controller
        self.last_v_error = 0
        self.last_w_error = 0
        self.TAM = np.array([[1.0, 1.0],
                             [0.0, 0.0], 
                             [self.thrusters_separation, -self.thrusters_separation]])
      
    def update(self, action):
        
        h = self.state[DOF.H]
        vx = self.state[DOF.Vx]
        vy = self.state[DOF.Vy]
        w = self.state[DOF.W]
        act = np.clip(action, -1, 1)
        
        # Compute dynamics
        if act[0] >= 0 and act[1] >= 0:     # Forward
            tau = np.min(act) * self.thrust
        elif act[0] < 0 and act[1] < 0:     # Bacward
            tau = np.max(act) * self.thrust / 3.0
        else:                               # Turn
            tau = 0

        # Get water current in vehicle coordinates
        wc_v = self.get_current_in_vehicle_frame()
        
        # Surge
        damping = self.linear_damping * vx + self.quadratic_damping * (abs(vx) * vx)
        f = tau - damping
        acc = f / self.mass
        acc = np.clip(acc, -5, 5)
        vx += self.dt * acc
        vx = np.clip(vx, -1.0, 1.0)
        
        # Sway
        damping = self.linear_damping * vy + self.quadratic_damping * (abs(vy) * vy)
        f = -damping
        acc = f / (self.mass * 2)
        acc = np.clip(acc, -5, 5)
        vy += self.dt * acc
        vy = np.clip(vy, -1.0, 1.0)
        
        # Turn 
        tau_diff = (act[0] - act[1]) * self.thrust * self.thrusters_separation
        damp_diff = self.linear_damping * w + self.quadratic_damping * (abs(w) * w)
        f_diff = tau_diff - damp_diff
        acc_diff = f_diff / self.mass
        acc_diff = np.clip(acc_diff, -1, 1)
        w += self.dt * acc_diff
        w = np.clip(w, -0.5, 0.5)

        # Update auv state
        self.state[DOF.X] += (vx + wc_v[0]) * self.dt * math.cos(h) - (vy + wc_v[1]) * self.dt * math.sin(h) 
        self.state[DOF.Y] += (vx + wc_v[0]) * self.dt * math.sin(h) + (vy + wc_v[1]) * self.dt * math.cos(h)
        self.state[DOF.H] = wrap_angle(self.state[DOF.H] + w * self.dt)
        self.state[DOF.Vx] = vx
        self.state[DOF.Vy] = vy
        self.state[DOF.W] = w

    def velocity_controller(self, desired_velocity):
        # Try ro reach v desired and w desired using a PD controller
        Pv = 100.0
        Pw = 100.0
        Dv = 1.0
        Dw = 1.0

        v_d, w_d = desired_velocity
        v = self.state[DOF.Vx]
        w = self.state[DOF.W]
        v_error = v_d - v
        w_error = w_d - w
        tau_v = Pv * v_error + Dv * (v_error - self.last_v_error)   
        tau_w = Pw * w_error + Dw * (w_error - self.last_w_error)
        if abs(tau_w) > 10: # priorize w control
            tau_v = 0
        self.last_v_error = v_error
        self.last_w_error = w_error

        # using a thruster allocation matrix to compute the thruster forces
        thrust = np.array([tau_v, 0, tau_w])   
        act = thrust @ self.TAM  
        return act
    
    def get_current_in_vehicle_frame(self):
        h = self.state[DOF.H]
        R = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
        return R.transpose().dot(self.wc)
                

