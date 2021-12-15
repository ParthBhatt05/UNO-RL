import numpy as np
import tqdm
import os
import pickle
import random
import matplotlib.pyplot as plt
from agent import UnoAgent
from environment import UnoEnvironment

PLAYER_COUNT = 2
COLLECTOR_THREADS = 2
INITIAL_EPSILON = 1
EPSILON_DECAY = 0.999999
MIN_EPSILON = 0.01
MODEL_PATH = 'example_model.h5'

from keras.models import load_model
model = load_model(MODEL_PATH)

def run(agent, episodes):
    # initialize environment
    epsilon = INITIAL_EPSILON
    players = [0, 1]
    
    env = UnoEnvironment(len(players))
    sa_pairs = set()
    model_wins = np.zeros(episodes)
    model_percent_win = np.zeros(episodes)
    
    episode_steps = np.zeros(episodes)
    episode_pairs = np.zeros(episodes)
    rewards = np.zeros(episodes)
    for e in tqdm.tqdm(range(episodes)):
        done = False
        state = None
        current = 0
        
        # run one episode
        while not done:
            episode_steps[e] += 1
            episode_pairs[e] = len(sa_pairs)
            if players[env.turn] == 0:
                current = 100
                state = env.get_state()
                action = np.argmax(model.predict(state.reshape((1, -1)))[0])

                # make random move if the AI selected an illegal move
                if not env.legal_move(action):
                    while not env.legal_move(action):
                        action = np.random.randint(env.action_count())

            elif players[env.turn] == 1:
                current = 0
                state = env.get_state()
                action = np.argmax(model.predict(state.reshape((1, -1)))[0])

                # make random move if the AI selected an illegal move
                if not env.legal_move(action):
                    while not env.legal_move(action):
                        action = np.random.randint(env.action_count())
            
            new_state, reward, done, _ = env.step(action)
            state = new_state
            
        # reset the environment for the next episode
        if current == 100:
            if e == 0:
                model_wins[e] = 1
                model_percent_win[e] = 1
            else:
                model_wins[e] = model_wins[e - 1] + 1
        else:
            model_wins[e] = model_wins[e - 1]
        model_percent_win[e] = model_wins[e] / (e + 1)
        env.reset()
    
    return rewards, episode_steps, model_wins, model_percent_win, episode_pairs

trials = 1
episodes = 1000

env = UnoEnvironment(2)
agent = UnoAgent(env.state_size(), env.action_count())

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for trial in range(1, trials + 1):
    trial_rewards, trial_steps, trial_winners, total_percent_win, trial_pairs = run(agent, episodes)

x = [0, 1000]
y = [0.5, 0.5]

percent_x = np.linspace(1, (len(total_percent_win)), len(total_percent_win))
percent_y = total_percent_win
percent_err = 1.96 * (np.std(percent_y) / np.sqrt(len(percent_y)))

pairs_x = np.linspace(1, (len(trial_pairs)), len(trial_pairs))
pairs_y = trial_pairs
pairs_err = 1.96 * (np.std(pairs_y) / np.sqrt(len(pairs_y)))

ax1.plot(total_percent_win)
ax1.plot(x, y, '--')
ax1.fill_between(percent_x-1, (percent_y - percent_err), (percent_y + percent_err),
                 alpha=0.2)
ax1.set_ylim([0.3, 0.7])
ax1.set_xlabel('episodes')
ax1.set_ylabel('Percent Win')
ax1.set_title('Percent wins for DQN vs DQN')
fig1.savefig('first_player_adv.png')

ax2.plot(trial_pairs)
ax2.fill_between(pairs_x-1, (pairs_y - pairs_err), (pairs_y + pairs_err),
                 alpha=0.2)
ax2.set_xlabel('episodes')
ax2.set_ylabel('state / action pairs')
ax2.set_title('State / action pair coverage for DQN')
fig2.savefig('dqn_coverage.png')


with open("win_dqn_data.pkl", "wb") as outfile1:
    pickle.dump(trial_winners, outfile1)
with open("reward_dqn_data.pkl", "wb") as outfile2:
    pickle.dump(trial_rewards, outfile2)
with open("steps_dqn_data.pkl", "wb") as outfile3:
    pickle.dump(trial_steps, outfile3)
with open("pairs_dqn_data.pkl", "wb") as outfile4:
    pickle.dump(trial_pairs, outfile4)
with open("percent_win_data.pkl", "wb") as outfile5:
    pickle.dump(total_percent_win, outfile5)
