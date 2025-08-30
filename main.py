from game import Pong
from agent import Agent

#env = Pong(render_mode="human", player1="human", player2="human", bot_difficulty="easy")
#env = Pong(render_mode="human", player1="human", player2="bot", bot_difficulty="easy")
#env = Pong(render_mode="human", player1="human", player2="bot", bot_difficulty="hard")

agent = Agent(eval=True, hidden_layer=756)

#env = Pong(render_mode="human", player1="human", player2="ai", bot_difficulty="easy", ai_agent=agent)
#env = Pong(render_mode="human", player1="ai", player2="human", bot_difficulty="easy", ai_agent=agent)
env = Pong(render_mode="human", player1="ai", player2="bot", bot_difficulty="hard", ai_agent=agent)

env.game_loop()
