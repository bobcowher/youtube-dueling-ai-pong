source ~/anaconda3/etc/profile.d/conda.sh

conda activate dueling_ai_pong

#export REPLAY_BUFFER_MEMORY="cpu"

python ./train.py
