from game import Pong

env = Pong(render_mode="human", player1="human", player2="human", bot_difficulty="easy")

env.game_loop()
