import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt
from gym_auv.auv2d_wc import AUV2D_WC, DOF, wrap_angle
from PIL import Image
import io


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Dockingv2(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 25}

    def __init__(self, render_mode='rgb_array', options={"continuous_actions": True, "max_entry_speed": 0.3, "max_entry_error": 0.5, "max_current": 0.3, "usbl_error": 0.025, "vel_error": 0.03}):
        super(Dockingv2, self).__init__()  
        self.render_mode = render_mode
        # 2D Docking - Vehicle Dinamics (forward/turn)
        # GOAL: Starting at ~30m from the DS, with random orientation and velocities, reaches the docking point with v < 0.3 
        # Basic AUV dynamics (dumping) and ocean current are considered.
        # The rewards function is based on:
        # * Dense reward according to distance to DS and energy utilization.
        # * If close to DS penalize speeds greater than desired docking speed (i.e. 0.3m/s).
        # * If AUV inside DS, reward according to J.Esteba 2021 paper and terminate.
        # * If collision with DS, negative reward and episode finalizes.

        # Continuous Observation: [dist_to_DS, angle_to_DS, auv_angle, auv_velocity_x, auv_velocity_y, auv_velocity_yaw (wrt DS)]
        # Continuous Actions: velocity [v, w] or predefined actions [forward, turn].
        # Max Steps: 600

        self.steps = 0
        self.max_steps = 600
        self.max_distance = 31
        self.max_entry_speed = options["max_entry_speed"] # 0.3
        self.max_entry_error = options["max_entry_error"] # 0.5
        self.max_current = np.clip(options["max_current"], 0, 0.5) # ~0.3
        self.continuous_actions = options["continuous_actions"]
        self.usbl_error = options["usbl_error"] # 0.05
        self.vel_error = options["vel_error"] # 0.1

        # Define action and observation space. They must be gymnasium.spaces objects.
        if self.continuous_actions:
            self.action_space = spaces.Box(np.array([-0.4, -0.3], dtype=np.float32), np.array([0.8, 0.3], dtype=np.float32))
        else:
            self.action_space = gym.spaces.Discrete(12) #forward: -0.3, 0, 3, 0.6, turn: -0.25, 0, 0.25 -> 4*3 = 12
            self.predefined_actions = []
            for v in [-0.3, 0, 0.35, 0.7]:
                for w in [-0.25, 0, 0.25]:
                    self.predefined_actions.append([v, w])

        # DS distance, DS bearing, AUV angle, AUV vx, AUV vy, AUV w # TODO: Introduce in observations previous action?
        self.observation_space = spaces.Box(np.array([0, -3.15, -3.15, -1.5, -1.5, -0.5], dtype=np.float32), np.array([500, 3.15, 3.15, 1.5, 1.5, 0.5], dtype=np.float32))

        # Reset state
        self.last_obs, _ = self.reset()

    def step(self, action):
        truncated = False
        terminated = False

        self.steps += 1
        
        # ------ Execute one time step within the environment ------
        if self.continuous_actions:
            desired_velocity = action
        else:
            desired_velocity = self.predefined_actions[action]

        tau = self.auv.velocity_controller(desired_velocity) # transform AUV desired speed (v, w) into thuster forces (tau)
        self.auv.update(tau) # update AUV dynamic model (inc_t = 0.2)
        
        # get current obs 
        self.last_obs = self.get_obs() # get current state

        #  ------ Compute Reward and check if Done ------
        # Penalize distance to goal
        dist = np.linalg.norm(self.auv.state[0:2])
        reward =  dist/self.max_distance * -1.0

        # Penalize energy used
        reward += -np.linalg.norm(tau)/100

        # If close to DS penalize difference in desired docking speed
        v_wc = self.auv.get_current_in_vehicle_frame()
        auv_vel_norm = np.linalg.norm(np.array([self.auv.state[DOF.Vx] + v_wc[0], self.auv.state[DOF.Vy] + v_wc[1]]))

        if dist < 3:
            velocity_penalization = max(auv_vel_norm - self.max_entry_speed, 0) * -10 * (3 - dist)/3
            if velocity_penalization > 0:
                print("Penalizing velocity: {}".format(velocity_penalization))
                print("auv_vel_norm: {}".format(auv_vel_norm))
                print("dist: {}".format(dist))
            reward += velocity_penalization

        # ------ Check terminal condition ------
        auv_x = self.auv.state[0]
        auv_y = self.auv.state[1]
        if auv_x >= -0.8 and auv_x <= 0.0 and abs(auv_y) <= self.max_entry_error: # inside DS
            terminated = True
            # Geometrical Analysis
            alpha_0 = normalize_angle(np.arctan2(auv_y, auv_x) - np.pi)
            e_angle = self.auv.state[2] - alpha_0
            w = 3.05
            if np.sign(e_angle) == np.sign(auv_y):
                w = 1
            g = 1 - w * abs(e_angle) * (1 + abs(alpha_0))
            reward = 200 * g
            # Penalize velocity --> TODO: Descomentar això!!!
            velocity_penalization = max(auv_vel_norm - self.max_entry_speed, 0) * -200
            reward = np.clip(reward + velocity_penalization, -200, 200)
            
        # Finalize episode if max_steps done
        if self.steps >= self.max_steps:
            truncated = True
        
        return self.last_obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options={}):
        # Increase environmet difficulty according to episode to make learning faster
        # print("Episode {}".format(self.episode))
        self.steps = 0
        self.auv = self.reset_state()
        obs = self.get_obs()
        info = {"initial_state": self.auv.state, "water_current": self.auv.wc}
        return (obs, info)
    
    def reset_state(self):
        # Reset the state of the environment to an initial state    
        init_auv = np.zeros(8)
        init_auv[0] = np.random.uniform(-30.0, -25.0)                           # x --> [-30, -25]
        init_auv[1] = np.random.uniform(-7.5, 7.5)                              # y --> [-7.5, 7.5]
        init_auv[2] = np.random.uniform(-np.pi/2, np.pi/2)                      # h --> [-pi/3, pi/3] -> [-60º, 60º]
        init_auv[3] = 0.0                                                       # vx = 0
        init_auv[4] = 0.0                                                       # vy = 0
        init_auv[5] = 0.0                                                       # w = 0  
        init_auv[6] = np.random.uniform(-self.max_current, self.max_current)    # wc_x --> [-max_current, max_current]
        init_auv[7] = np.random.uniform(-self.max_current, self.max_current)    # wc_y --> [-max_current, max_current]
        return AUV2D_WC(init_auv)

    def get_obs(self):
        obs = np.zeros(6, dtype=np.float32)

        # Polar position to DS
        dist_to_target = np.linalg.norm(self.auv.state[0:2])
        angle_to_target = np.arctan2(-self.auv.state[DOF.Y], -self.auv.state[DOF.X]) - self.auv.state[DOF.H]

        # get current speed in AUV frame
        v_wc = self.auv.get_current_in_vehicle_frame()

        # Add noise to observations. WARNING! heading noise is hardcoded to 0.05
        obs[0] = dist_to_target * np.random.normal(1, self.usbl_error)                                                               # Distance to DS from AUV
        obs[1] = angle_to_target + np.random.normal(0, self.usbl_error)                                                              # Angle to DS RELATIVE TO AUV
        obs[2] = self.auv.state[DOF.H] + np.random.normal(0, 0.05)                                                                   # AUV angle
        obs[3] = (self.auv.state[DOF.Vx] + v_wc[0] + np.random.normal(0, self.vel_error/2))                                          # AUV velocity in x wrt floor
        obs[4] = (self.auv.state[DOF.Vy] + v_wc[1] + np.random.normal(0, self.vel_error/2))                                          # AUV velocity in y wrt floor
        obs[5] = (self.auv.state[DOF.W]  + np.random.normal(0, self.vel_error/10)) * np.random.normal(1, self.vel_error)             # AUV angular speed
        
        return obs

    def render(self, close=False):
        # Render the environment to the screen
        plt.clf()
        
        # Plot DS
        plt.axis([-self.max_distance, self.max_distance, -self.max_distance, self.max_distance])
        plt.plot([0, 1, 2, 2, 1, 0], [0.65, 0.2, 0.2, -0.2, -0.2, -0.65], 'k')
        
        # Plot AUV
        plt.arrow(self.auv.state[0] - np.cos(self.auv.state[2])*0.8, self.auv.state[1] - np.sin(self.auv.state[2])*0.8, 1.6*np.cos(self.auv.state[2]), 1.6*np.sin(self.auv.state[2]), head_width=0.2, head_length=0.3, fc='y', ec='y')
        
        # Plot current
        plt.arrow(-25, -7.5, 10*self.auv.wc[0], 10*self.auv.wc[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
        
        plt.xlim(-30, 5)
        plt.ylim(-10, 10)
    
        if self.render_mode == "rgb_array":
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=200)
            buf.seek(0)

            # Load the image into a PIL Image object
            img = Image.open(buf)

            # Convert the image into an RGB matrix
            rgb_matrix = np.array(img)
            return rgb_matrix[:,:,0:3]
        
        plt.pause(0.001)