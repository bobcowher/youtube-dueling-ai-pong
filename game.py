import pygame
import sys
import gymnasium as gym
import os
import time
from assets import *
import cv2
import numpy as np
import torch

class Pong(gym.Env):

    def __init__(self, window_width=1280, window_height=960, fps=60, player1="ai", player2="bot",
                 render_mode="rgb_array", step_repeat=4, bot_difficulty="hard", ai_agent=None):

        for p in [player1, player2]:
            if p not in {"ai", "bot", "human"}:
                raise ValueError(f"All players must be ai, bots, or humans")

        self.window_width = window_width
        self.window_height = window_height
        self.step_repeat = step_repeat
        self.render_mode = render_mode
        
        if(self.render_mode != "human"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.display.set_caption("Pong")

        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.fps = fps

        self.player_1_color = (50, 205, 50)
        self.player_2_color = (138, 43, 226)

        self.background_color = (0, 0, 0)

        self.paddle_height = 120
        self.paddle_width = 20

        self.bot_difficulty = bot_difficulty

        self.font = pygame.font.SysFont(None, 70)
        self.announcement_font = pygame.font.SysFont(None, 150)

        self.player1 = player1
        self.player2 = player2
        
        self.action_space = gym.spaces.Discrete(3)

        self.ai_agent = ai_agent

        print("Creating new Pong game")
        print("Players:")
        print("Player 1: ", player1)
        print("Player 2: ", player2)
        print("Bot difficulty: ", self.bot_difficulty)

        self.reset()

    def reset(self):
        self.player_1_score = 0
        self.player_2_score = 0

        self.top_score = 20

        self.player_1_paddle = Paddle(x=self.window_width - 2 * (self.window_width / 64),
                                      y=(self.window_height / 2) - (self.paddle_height / 2),
                                      player_color=self.player_1_color, 
                                      height=self.paddle_height,
                                      width=self.paddle_width,
                                      window_height=self.window_height);
        
        self.player_2_paddle = Paddle(x=(self.window_width / 64),
                                      y=(self.window_height / 2) - (self.paddle_height / 2),
                                      player_color=self.player_2_color, 
                                      height=self.paddle_height,
                                      width=self.paddle_width,
                                      window_height=self.window_height);

        self.ball = Ball(window_height=self.window_height,
                         window_width=self.window_width, 
                         height=20,
                         width=20,
                         player_1_paddle=self.player_1_paddle,
                         player_2_paddle=self.player_2_paddle)

        return self._get_obs(), {}
        

    def _get_obs(self):
        screen_array = pygame.surfarray.pixels3d(self.screen)

        screen_array = np.transpose(screen_array, (1, 0, 2))

        downscaled_image = cv2.resize(screen_array, (84, 84), interpolation=cv2.INTER_NEAREST)

        grayscale = cv2.cvtColor(downscaled_image, cv2.COLOR_RGB2GRAY)

        grayscale[grayscale != 0] = 255

        observation = torch.from_numpy(grayscale).float().unsqueeze(0)

        return observation


    def fill_background(self):
        self.screen.fill(self.background_color)

        player_1_score_surface = self.font.render(f'Score: {self.player_1_score}', 
                                                  True,
                                                  self.player_1_color)
        
        player_2_score_surface = self.font.render(f'Score: {self.player_2_score}', 
                                                  True,
                                                  self.player_2_color)

        self.screen.blit(player_1_score_surface, ((self.window_width / 2) + 20, 10))
        self.screen.blit(player_2_score_surface, ((self.window_width / 2) - player_2_score_surface.get_width() - 20, 10))


    def game_over(self):
        if(self.player_1_score >= self.top_score):
            game_over_surface = self.announcement_font.render('Player 1 Won', 
                                                              True,
                                                              self.player_1_color)
        if(self.player_2_score >= self.top_score):
            game_over_surface = self.announcement_font.render('Player 2 Won', 
                                                              True,
                                                              self.player_2_color)
        game_over_rect = game_over_surface.get_rect(center=(self.window_width // 2,
                                                    self.window_height // 2))

        self.screen.blit(game_over_surface, game_over_rect)

        pygame.display.flip()

        self.done = True

        time.sleep(10)

        pygame.quit()
        sys.exit()



    def game_loop(self):
        
        while(True):
        
            player_1_action = 0
            player_2_action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            keys = pygame.key.get_pressed()

            if self.player1 == "human":
                if keys[pygame.K_k]:
                    player_1_action = 1
                elif keys[pygame.K_j]:
                    player_1_action = 2
            if self.player2 == "human":
                if keys[pygame.K_e]:
                    player_2_action = 1
                elif keys[pygame.K_f]:
                    player_2_action = 2

            
            if self.player1 == "bot":
                player_1_action = self.get_bot_move(player=1)
            if self.player2 == "bot":
                player_2_action = self.get_bot_move(player=2)
            

            observation, player_1_reward, player_2_reward, done, truncated, info = self.step(player_1_action=player_1_action,
                      player_2_action=player_2_action)

            if(player_1_reward != 0):
                print("Player 1 reward:", player_1_reward)
                print("Player 2 reward:", player_2_reward)
                print("Done: ", done)


    def get_bot_move(self, player):
        
        random_target = 0.1
        next_move = 0

        player_y = self.player_1_paddle.y if player == 1 else self.player_2_paddle.y

        if(self.bot_difficulty == "easy"):
            if random.random() <= random_target:
                next_move = random.randint(0, 2)
            else:
                if(self.ball.vy > 0):
                    next_move = 2
                else: 
                    next_move = 1
        elif(self.bot_difficulty == "hard"):
            if random.random() <= random_target:
                next_move = random.randint(0, 2)
            else:
                if(self.ball.y > (player_y + 60)):
                    next_move = 2
                elif(self.ball.y < (player_y + 60)):
                    next_move = 1

        return next_move



    def step(self, player_1_action=None, player_2_action=None):

        player_1_reward = 0
        player_2_reward = 0
        info = {}
        done = False
        truncated = False

        if(player_1_action is None):
            player_1_action = self.get_bot_move(1)
        if(player_2_action is None):
            player_2_action = self.get_bot_move(2)

        for i in range(self.step_repeat):
            self._step(player_1_action=player_1_action,
                       player_2_action=player_2_action)

        observation = self._get_obs()

        ball_center = self.ball.x + (self.ball.width / 2)

        if(ball_center < 0):
            self.player_1_score += 1
            player_1_reward += 1
            player_2_reward -= 1
            self.ball.spawn()
        elif(ball_center > self.window_width):
            self.player_2_score += 1
            player_1_reward -= 1
            player_2_reward += 1
            self.ball.spawn()

        if(self.player_1_score >= self.top_score or
           self.player_2_score >= self.top_score):
            if self.render_mode == "human":
                self.game_over()
            else:
                done = True
                truncated = True

        return observation, player_1_reward, player_2_reward, done, truncated, info

    
    def _step(self, player_1_action=0, player_2_action=0):
        
        self.player_1_paddle.move(player_1_action)
        self.player_2_paddle.move(player_2_action)

        self.fill_background()

        self.player_1_paddle.draw(screen=self.screen)
        self.player_2_paddle.draw(screen=self.screen)
        self.ball.move()
        self.ball.draw(screen=self.screen)

        if(self.render_mode == "human"):
            self.clock.tick(self.fps)
            pygame.display.flip()


    



