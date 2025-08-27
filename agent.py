import torch
from torch._C import device
import torch.optim as optim
import torch.nn.functional as F
from buffer import ReplayBuffer
from model import Model
from game import Pong
from collections import deque
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import random

class Agent():

    def __init__(self, hidden_layer=512,
                       learning_rate=0.0001,
                       gamma=0.99,
                       max_buffer_size=200000,
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

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, 
                                   n_actions=self.env.action_space.n, input_device=self.device,
                                   output_device=self.device) 

        self.model = Model(action_dim=3, hidden_dim=hidden_layer, observation_shape=obs.shape, obs_stack=self.frame_stack).to(self.device)
        self.target_model = Model(action_dim=3, hidden_dim=hidden_layer, observation_shape=obs.shape, obs_stack=self.frame_stack).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        self.max_episode_steps = max_episode_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
  


    def train(self, episodes, summary_writer_suffix, batch_size):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'

        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        total_steps = 0

        best_avg_score = -100

        for episode in range(episodes):
            done = False
            player_1_episode_reward = 0
            player_2_episode_reward = 0

            obs, info = self.env.reset()

            obs = self.process_observation(obs, clear_stack=True)

            episode_steps = 0

            episode_start_time = time.time()
            
            while not done and episode_steps < self.max_episode_steps:

                player_1_action = self.get_action(obs, player=1, checkpoint_model=False, episode=episode)

                player_1_reward = 0
                player_2_reward = 0
                
                next_obs, player_1_reward, player_2_reward, done, truncated, info = self.env.step(player_1_action=player_1_action)

                next_obs = self.process_observation(next_obs)

                self.memory.store_transition(obs, player_1_action, player_1_reward, next_obs, done)

                obs = next_obs

                player_1_episode_reward += player_1_reward
                player_2_episode_reward += player_2_reward
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):

                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

                    actions = actions.unsqueeze(1).long()
                    rewards = rewards.unsqueeze(1)
                    dones = dones.unsqueeze(1).float()

                    q_values = self.model(observations)
                    q_sa     = q_values.gather(1, actions)

                    with torch.no_grad():
                        next_actions = torch.argmax(
                            self.model(next_observations), dim=1, keepdim=True
                        )

                        next_q = self.target_model(next_observations).gather(1, next_actions)
                        targets = rewards + (1 - dones) * self.gamma * next_q

                    loss = F.mse_loss(q_sa, targets)
                    writer.add_scalar("Stats/model_loss", loss.item(), total_steps)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if total_steps % self.target_update_interval == 0:
                        self.target_model.load_state_dict(self.model.state_dict())


            writer.add_scalar('Stats/Epsilon', self.epsilon, episode)

            if(self.epsilon > self.min_epsilon):
                self.epsilon *= self.epsilon_decay

            writer.add_scalar("Training Score/Player 1 Training", player_1_episode_reward, episode)
            writer.add_scalar("Training Score/Player 2 Training", player_2_episode_reward, episode)

            episode_end_time = time.time()
            episode_time = episode_end_time - episode_start_time

            print(f"Completed episode {episode} with Player 1 score {player_1_episode_reward}")
            print(f"Completed episode {episode} with Player 2 score {player_2_episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")
            
                


    def get_action(self, obs, player=2, checkpoint_model=False, episode=0, eval_mode=False):

    # TODO: Flip obs if player 2
    # TODO: Checkpoint Model
        if(random.random() < self.epsilon) and (eval_mode == False):
            action = self.env.action_space.sample()
        else:
            q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
            action = torch.argmax(q_values, dim=-1).item()

        return action


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

