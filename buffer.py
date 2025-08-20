import torch
import numpy as np
import os

class ReplayBuffer:

    def __init__(self, max_size, input_shape, n_actions,
                 input_device, output_device='cpu', frame_stack=3):
        
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")

        if override in ["cpu", "cuda:0", "cuda:1"]:
            print(f"Received replay buffer memory override to {override}")
            self.input_device = override
        else:
            self.input_device = input_device

        self.output_device = output_device

        self.state_memory = torch.zeros((max_size, *input_shape), dtype=torch.uint8, device=self.input_device)
        self.next_state_memory = torch.zeros((max_size, *input_shape), dtype=torch.uint8, device=self.input_device)

        self.action_memory = torch.zeros(max_size, dtype=torch.int64, device=self.input_device)
        self.reward_memory = torch.zeros(max_size, dtype=torch.float32, device=self.input_device)
        self.terminal_memory = torch.zeros(max_size, dtype=torch.bool, device=self.input_device)


    def can_sample(self, batch_size: int) -> bool:
        return self.mem_ctr >= batch_size * 10
    

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_ctr % self.mem_size

        self.state_memory[idx] = torch.as_tensor(state, dtype=torch.uint8, device=self.input_device)
        self.next_state_memory[idx] = torch.as_tensor(next_state, dtype=torch.uint8, device=self.input_device)

        self.action_memory[idx] = int(action)
        self.reward_memory[idx] = float(reward)
        self.terminal_memory[idx] = bool(done)
        
        self.mem_ctr += 1

        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        
        batch   = torch.randint(0, max_mem, (batch_size,), device=self.input_device, dtype=torch.int64)

        states      = self.state_memory[batch].to(self.output_device, dtype=torch.float32)
        next_states = self.next_state_memory[batch].to(self.output_device, dtype=torch.float32)
        rewards     = self.reward_memory[batch].to(self.output_device)
        dones       = self.terminal_memory[batch].to(self.output_device)
        actions     = self.action_memory[batch].to(self.output_device)

        return states, actions, rewards, next_states, dones





