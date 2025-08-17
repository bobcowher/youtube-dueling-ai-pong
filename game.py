import pygame
import sys
import gymnasium as gym
import os

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

            if(player_1_action != 0):
                print("Player 1 Action: ", player_1_action)
            
            if(player_2_action != 0):
                print("Player 2 Action: ", player_2_action)

            self.step()


    def step(self, player_1_action=None, player_2_action=None):
        self.fill_background()
        
        if(self.render_mode == "human"):
            self.clock.tick(self.fps)
            pygame.display.flip()
    



