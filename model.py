from sys import exc_info
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, action_dim, hidden_dim=256, observation_shape=None, obs_stack=4) -> None:
        super(Model, self).__init__()

        # 3 Conv Layers
        # Input - obs_stack
        # out_channels 32, kernal size 8, stride 4
        # 32, 64, k=4, s=2
        # 64, 64, k=3, s=1

        self.conv1 = nn.Conv2d(in_channels=obs_stack, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv_output_size = self.calculate_conv_output(observation_shape)

        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

        self.apply(self.weights_init)

    def calculate_conv_output(self, observation_shape):
        x = torch.zeros(1, *observation_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x.view(-1).shape[0]


    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = x / 255

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) # flatten.

        x = F.relu(self.fc1(x))

        output = self.output(x)

        return output


    def save_the_model(self, filename='models/latest.pt'):
        torch.save(self.state_dict(), filename)

    
    def load_the_model(self, filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(filename))
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}")

    
        

