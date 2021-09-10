import gym
from gym import spaces
import airsim
import numpy as np
import time
import math

class CarGym(gym.Env):
    def __init__(self):
        self.airsim_client = airsim.CarClient()
        self.airsim_client.confirmConnection()
        self.airsim_client.enableApiControl(True)
        print("API Control enabled: %s" % self.airsim_client.isApiControlEnabled())
        self.car_controls = airsim.CarControls()
        self.new_break_val = 0
        self.airsim_client.reset()

        self.state = {"position": np.zeros(3), "collision": None, "prev_position": np.zeros(3), "pose": None,
                      "prev_pose": None}

        # CarControls are:
        # throttle  0.0
        # steering = 0.0
        # brake = 0.0
        # handbrake = False
        # is_manual_gear = False
        # manual_gear = 0
        # gear_immediate = True
        # so action is the 3-element vector which increases/decreases ONLY throttle, steering, and brake.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box( low=0, high=255, shape=(1, 84, 84))

        # create an image request from camera "0", image type is depth perspective, pixels as float = True, compress= False
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

    def _do_action(self, action):
        """
        transfoms action into carControls and feeds it into the airsim client(). Then it goes to sleep.
        :param action: a 7-element array, where each element increases/decreases the corresponding carControl() object
        :return: None
        """
        # throttle  0.0
        new_throttle = self.car_controls.throttle + action[0]
        if new_throttle >= 1.0:
            new_throttle = 1.0
        elif new_throttle < -1.0:
            new_throttle = -1.0
        self.car_controls.throttle = new_throttle

        # steering = 0.0
        new_steering = self.car_controls.steering + action[1]
        if new_steering >= 1.0:
            new_steering = 1.0
        elif new_steering <=-1.0:
            new_steering = -1.0
        self.car_controls.steering = new_steering

        # brake = 0.0
        if  action[2] >= 0.5:
            self.car_controls.brake = True
        else:
            self.car_controls.brake = False
        car_controls = airsim.CarControls()
        car_controls.throttle = self.car_controls.throttle
        car_controls.steering = self.car_controls.steering
        # car_controls.brake = self.car_controls.brake

        self.airsim_client.setCarControls(car_controls)

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        #img1d is an ndarray of size (36864,). min=1.986 max=65504.0
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        #img1d is now an ndaray of size (36864,) min=0.00389 max=128.377
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        #img2d is ndarray(144,256)
        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))
        # im_final is ndarray(84,84)
        # image is now resized and converted to gray scale.
        return_im = im_final.reshape([1, 84, 84])
        # return_im is #ndarray(84,84,1)
        return return_im

    def _get_obs(self):
        responses = self.airsim_client.simGetImages([self.image_request])
        image = self.transform_obs(responses)

        self.car_state = self.airsim_client.getCarState()
        collision = self.airsim_client.simGetCollisionInfo().has_collided

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = collision
        return image

    def _compute_reward(self):
        """
         reward is computed as reward_dist + reward_speed.
                reward_dist is either:
                    (-3), when the car's smallest distance to a set of points is bigger than a threshold.
                    (math.exp( -3 * dist) - 0.5) (if the dist is large, reward is small) otherwise
                reward_speed is:
                    (-) if the car's speed is below halfway of MIN_SPEED and MAX_SPEED
                    (0) if the car's speed is exactly halfway
                    (+) if the car's speed is above halfway
        on the other hand, the episode is done if:
                the reward is < 1
                brake's are off, but speed < 1
                there's a collision
        :return: reward, done
        """
        MAX_SPEED = 300
        MIN_SPEED = 10
        thresh_dist = 3.5
        beta = 3

        z = 0
        pts = [
            np.array([0, -1, z]),
            np.array([130, -1, z]),
            np.array([130, 125, z]),
            np.array([0, 125, z]),
            np.array([0, -1, z]),
            np.array([130, -1, z]),
            np.array([130, -128, z]),
            np.array([0, -128, z]),
            np.array([0, -1, z]),
        ]
        pd = self.state["pose"].position
        car_pt = np.array([pd.x_val, pd.y_val, pd.z_val])

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i + 1])))
                / np.linalg.norm(pts[i] - pts[i + 1]),
                )

        # print(dist)
        if dist > thresh_dist:
            reward = -3
        else:
            reward_dist = math.exp(-beta * dist) - 0.5

            reward_speed = (
                                   (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
                           ) - 0.5
            reward = reward_dist + reward_speed

        done = 0
        if reward < -1:
            print("Done because reward <-1")
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1

        return reward, done

    def _compute__reward(self):
        """Reward #1: Customized reward system for getting to the end of the road.

        :return: reward, done
        """
        reward = 0
        # target position
        targe_pos = {
            "x_val": 101.336,
            "y_val": 0.077,
            "z_val": -0.587

        }


        # Get the car's current position and previous position
        curr_pos = self.state["pose"].position
        prev_pos = self.state["prev_pose"].position

        done = 0
        if reward < -1:
            print("Done because reward <-1")
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            done = 1

        return reward, done


    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute__reward()

        return obs, reward, done, self.state

    def reset(self):
        self.airsim_client.reset()
        return self._get_obs()

    def render(self, mode='human'):
        pass

