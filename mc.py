import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle
from agent import UnoAgent
from environment import UnoEnvironment
from collections import defaultdict
from tqdm import trange

PLAYER_COUNT = 2
COLLECTOR_THREADS = 2
INITIAL_EPSILON = 1
EPSILON_DECAY = 0.999999
MIN_EPSILON = 0.01
MODEL_PATH = 'example_model.h5'

from keras.models import load_model
model = load_model(MODEL_PATH)

def make_epsilon_greedy(Q, epsilon, N):
    
    def policy_fn(observation):
        A = np.ones(N, dtype = float) * epsilon / N
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    
    return policy_fn

def run(agent, episodes, gamma=1):
    # initialize environment
    epsilon = MIN_EPSILON
    players = [0, 1]
    
    env = UnoEnvironment(len(players))
    sa_pairs = set()
    episode_steps = np.zeros(episodes)
    av_episode_steps = np.zeros(episodes)
    rewards = np.zeros(episodes)
    av_rewards = np.zeros(episodes)
    episode_pairs = np.zeros(episodes)
    model_wins = np.zeros(episodes)
    model_percent_win = np.zeros(episodes)
    
    Q = defaultdict(lambda: np.zeros(env.action_count()))
    N = defaultdict(lambda: np.zeros(env.action_count()))
    
    for e in tqdm.tqdm(range(episodes)):
        done = False
        current = 0
        G = 0
        seen = set()
        # run one episode
        while not done:
            episode_steps[e] += 1
            episode_pairs[e] = len(sa_pairs)
            if players[env.turn] == 0:
                current = 100
                state = str(env.get_state())
                
                policy = make_epsilon_greedy(Q, epsilon, env.action_count())
                action_prob = policy(state)
                action = np.random.choice(np.arange(len(action_prob)), p=action_prob)

                # make random move if the AI selected an illegal move
                
                while not env.legal_move(action):
                    action = np.random.randint(env.action_count())
                    
                new_state, reward, done, step_info = env.step(action)
                G = (gamma * G) + reward
                if state not in seen:
                    seen.add(state)
                    N[(str(state), action)] = np.append(N[(str(state), action)], G)
                    Q[(str(state), action)] = np.mean(N[(str(state), action)])
                    
                rewards[e] += reward
                
                if state is not None:
                    if (str(state), action) not in sa_pairs:
                        sa_pairs.add((str(state), action))

            elif players[env.turn] == 1:
                current = 0
                
                action = np.random.randint(env.action_count())
                while not env.legal_move(action):
                    action = np.random.randint(env.action_count())
            
                new_state, reward, done, step_info = env.step(action)
            
            state = new_state

        if current == 100:
            if e == 0:
                model_wins[e] = 1
                model_percent_win[e] = 1
            else:
                model_wins[e] = model_wins[e - 1] + 1
        else:
            model_wins[e] = model_wins[e - 1]
        model_percent_win[e] = model_wins[e] / (e + 1)
        
        rewards[e] = rewards[e] / episode_steps[e]
        
        if e != 0:
            rewards[e] += rewards[e - 1]
            episode_steps[e] += episode_steps[e - 1]
            
        av_rewards[e] = rewards[e] / (e + 1)
        av_episode_steps = episode_steps[e] / (e + 1)
        
        env.reset()
    
    return av_rewards, av_episode_steps, model_wins, episode_pairs, model_percent_win



trials = 1
episodes = 50000

env = UnoEnvironment(2)
agent = UnoAgent(env.state_size(), env.action_count())

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for trial in range(1, trials + 1):
    trial_rewards, trial_steps, trial_winners, trial_pairs, total_percent_win = run(agent, episodes)

percent_x = np.linspace(1, (len(trial_rewards)), len(trial_rewards))
percent_y = trial_rewards
percent_err = 1.96 * (np.std(percent_y) / np.sqrt(len(percent_y)))


ax1.plot(trial_rewards)
ax1.fill_between(percent_x-1, (percent_y - percent_err), (percent_y + percent_err),
                 alpha=0.2)
ax1.set_xlabel('episodes')
ax1.set_ylabel("Reward")
ax1.set_title('Reward for MC vs random')
fig1.savefig('mc_rewards.png')


with open("win_mc_data2.pkl", "wb") as outfile1:
    pickle.dump(trial_winners, outfile1)
with open("reward_mc_data2.pkl", "wb") as outfile2:
    pickle.dump(trial_rewards, outfile2)
with open("steps_mc_data2.pkl", "wb") as outfile3:
    pickle.dump(trial_steps, outfile3)
with open("pairs_mc_data2.pkl", "wb") as outfile4:
    pickle.dump(trial_pairs, outfile4)
with open("percent_win_mc_data2.pkl", "wb") as outfile5:
    pickle.dump(total_percent_win, outfile5)
