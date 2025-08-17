from game import Pong

#env = Pong(render_mode="human", player1="human", player2="human", bot_difficulty="easy")
#env = Pong(render_mode="human", player1="human", player2="bot", bot_difficulty="easy")
#env = Pong(render_mode="human", player1="human", player2="bot", bot_difficulty="hard")
env = Pong(render_mode="human", player1="bot", player2="human", bot_difficulty="easy")

env.game_loop()
