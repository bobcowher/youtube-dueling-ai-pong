import torch
from model import Model
from game import Pong
from collections import deque

class Agent():

    def __init__(self, hidden_layer=512,
                       learning_rate=0.0001,
                       gamma=0.99,
                       max_buffer_size=100000,
                       eval=False,
                       frame_stack=3,
                       target_update_interval=10000,
                       max_episode_steps=1000,
                       epsilon=0,
                       min_epsilon=0,
                       epsilon_decay=0.995,
                       checkpoint_pool=5
                       ):
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.frame_stack = frame_stack
        self.frames = deque(maxlen=self.frame_stack)

        
        self.env = Pong(player1="ai", player2="bot", render_mode="rgb_array", bot_difficulty="easy")

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        print(f"Creating agent with device {self.device}...")

        model = Model(action_dim=3, hidden_dim=hidden_layer, observation_shape=obs.shape, obs_stack=self.frame_stack)

    
    def init_frame_stack(self, obs):
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)


    def process_observation(self, obs, clear_stack=False):
        obs = torch.tensor(obs, dtype=torch.float32)

        if(len(self.frames) < self.frame_stack):
            self.init_frame_stack(obs)

        if(clear_stack):
            self.init_frame_stack(obs)

        self.frames.append(obs)

        obs_stacked = torch.cat(tuple(self.frames), dim=0)

        return obs_stacked

