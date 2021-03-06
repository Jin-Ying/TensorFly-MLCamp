import os,sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import gym
from gym import spaces
import torch
import os
import numpy as np
from find_closest_index import find_closest_index

from stable_baselines.common.vec_env import SubprocVecEnv, VecEnv, DummyVecEnv
img_dir = 'pets'
imgs = os.listdir(img_dir)
N_DISCRETE_ACTIONS = len(imgs)
HEIGHT = 256
WIDTH = 256
N_CHANNELS = 3
def rescale(image):
    w = image.size[0]
    h = image.size[1]
    random_size = 256
    if w < h:
        img = image.resize((random_size,round(h/w * random_size)))
    else:
        img = image.resize((round(w/h * random_size),random_size))
    w = image.size[0]
    h = image.size[1]
    img=img.crop(((w-256)/2,(h-256)/2,(w+256)/2,(h+256)/2))
    return img
pictures=[]
for img in imgs: #遍历，进行批量转换
    first_name, second_name = os.path.splitext(img)
    img = os.path.join(img_dir , img)
    img = Image.open(img)
    width, height = img.size # 获取图片尺寸
    pictures.append(rescale(img))
def get_input():
    inp = input('input the score of similarty: ')
    if inp:
        try:
            score = int(inp)
        except:
            print('Input is not correct, please try again')
            return get_input()
    return score
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.feat_model = torch.load("./snapshot/finetune_pet/iter_01000_model.pth.tar",map_location=torch.device('cpu'))[0]
        self.cls_model = torch.load(os.path.join("./snapshot/reid_bulldog_new", "iter_05000_classifier.pth.tar"),map_location=torch.device('cpu'))[0]
        self.img_list = np.load("./snapshot/reid_bulldog_new/image_list.npy")
        self.gallery = np.load(os.path.join("./snapshot/reid_bulldog_new/", "2499_feature.npy"))
        self.image = torch.rand(1,3,224,224)
        
    def step(self, action):
        # action: picture_id
        # observation: picture
        # reward: 12345->{-1,-0.5,0,0.5,1}
        feature,_=self.feat_model(self.image)
        feature,_=self.cls_model(feature)
        feature_npy = feature.detach().cpu().numpy()
        dis_index = find_closest_index(feature_npy[0], gallery, top_k=topk)
        idx = dis_index[0]
        observation = np.array(pictures[idx])
        info = {}
        score = get_input()
        # reward = score/2.0-1.5
        reward = 1 if score else -1
        done = 0
        if score == 5:
            done = 1
            print('Congratulations! You have found your pets.')
            self.close()
        return observation, reward, done, info
    def reset(self):
        idx = random.randint(0,len(pictures)-1)
        # idx = random.randint(0,len(pictures)-1)
        observation = np.array(pictures[idx])
        return observation  # reward, done, info can't be included
    def render(self, mode='human'):
        pass
    def close (self):
        pass
import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC,PPO2

# env = gym.make('Pendulum-v0')
env = CustomEnv()

env = DummyVecEnv([lambda: env])
model = PPO2('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("ppo2_pendulum")