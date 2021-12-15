import numpy as np
import tqdm
import os
import pickle
import random
import matplotlib.pyplot as plt
from agent import UnoAgent
from environment import UnoEnvironment

PLAYER_COUNT = 3
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
    players = [0, 1, 2]
    
    env = UnoEnvironment(len(players))
    sa_pairs = set()
    model_wins = np.zeros(episodes)
    model_percent_win = np.zeros(episodes)
    
    episode_steps = np.zeros(episodes)
    av_episode_steps = np.zeros(episodes)
    episode_pairs = np.zeros(episodes)
    rewards = np.zeros(episodes)
    av_rewards = np.zeros(episodes)
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
                if state is None or np.random.sample() < epsilon or not agent.initialized:
                    # choose a random action
                    action = np.random.randint(env.action_count())
                else:
                    # choose an action from the policy
                    action = agent.predict(state)

                while not env.legal_move(action):
                    action = np.random.randint(env.action_count())
                    
                new_state, reward, done, _ = env.step(action)
                rewards[e] += reward
                

                if state is not None:
                    # include the current transition in the replay memory
                    agent.update_replay_memory((state, action, reward, new_state, done))
                if (str(state), action) not in sa_pairs:
                    sa_pairs.add((str(state), action))
                state = new_state

                if agent.initialized:
                    # decay epsilon
                    epsilon *= EPSILON_DECAY
                    epsilon = max(epsilon, MIN_EPSILON)
                
                if len(agent.replay_memory) < agent.BATCH_SIZE:
                    # wait until enough data is collected
                    continue

                # get minibatch from replay memory
                minibatch = np.array(random.sample(agent.replay_memory, agent.BATCH_SIZE))

                # get states from minibatch
                states = np.array(list(minibatch[:,0]))
                # predict Q values for all states in the minibatch
                q_values = agent.model.predict(states)
                # estimate the maximum future reward
                max_future_q = np.max(agent.model.predict(np.array(list(minibatch[:,3]))), axis=1)

                for i in range(len(minibatch)):
                    action, reward, done = minibatch[i,1], minibatch[i,2], minibatch[i,4]

                    # update the Q value of the chosen action
                    q_values[i,action] = reward
                    if not done:
                        # add the discounted maximum future reward if the current transition was not the last in an episode
                        q_values[i,action] += agent.DISCOUNT_FACTOR * max_future_q[i]

                # train the model on the minibatch
                hist = agent.target_model.fit(x=states, y=q_values, batch_size=agent.BATCH_SIZE, verbose=0)
                agent.logger.scalar('loss', hist.history['loss'][0])
                agent.logger.scalar('acc', hist.history['acc'][0])
                agent.logger.flush()

                if episode_steps[e] % agent.MODEL_UPDATE_FREQUENCY == 0:
                    # update the predictor model
                    agent.model.set_weights(agent.target_model.get_weights())
                    if not agent.initialized:
                        print('Agent initialized')
                        agent.initialized = True

                if episode_steps[e] % agent.MODEL_SAVE_FREQUENCY == 0:
                    # create model folder
                    folder = f'models/{agent.logger.timestamp}'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    # save model
                    agent.model.save(f'{folder}/model-{episode_steps[e]}.h5')

            elif players[env.turn] == 1 or players[env.turn] == 2:
                current = 0
                action = np.random.randint(env.action_count())
                while not env.legal_move(action):
                    action = np.random.randint(env.action_count())
            
                new_state, reward, done, step_info = env.step(action)
                
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
        
        rewards[e] = rewards[e] / episode_steps[e]
        
        if e != 0:
            rewards[e] += rewards[e - 1]
            episode_steps[e] += episode_steps[e - 1]
            
        av_rewards[e] = rewards[e] / (e + 1)
        av_episode_steps = episode_steps[e] / (e + 1)
        
        env.reset()
    
    return av_rewards, av_episode_steps, model_wins, model_percent_win, episode_pairs

trials = 1
episodes = 50

env = UnoEnvironment(3)
agent = UnoAgent(env.state_size(), env.action_count())
del env

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for trial in range(1, trials + 1):
    trial_rewards, trial_steps, trial_winners, total_percent_win, trial_pairs = run(agent, episodes)

percent_x = np.linspace(1, (len(total_percent_win)), len(total_percent_win))
percent_y = total_percent_win
percent_err = 1.96 * (np.std(percent_y) / np.sqrt(len(percent_y)))

pairs_x = np.linspace(1, (len(trial_pairs)), len(trial_pairs))
pairs_y = trial_pairs
pairs_err = 1.96 * (np.std(pairs_y) / np.sqrt(len(pairs_y)))

ax1.plot(total_percent_win)
ax1.fill_between(percent_x-1, (percent_y - percent_err), (percent_y + percent_err),
                 alpha=0.2)
ax1.set_xlabel('episodes')
ax1.set_ylabel('Percent Win')
ax1.set_title('Percent wins for DQN vs two random')
fig1.savefig('dqn_wins_3p.png')

ax2.plot(trial_pairs)
ax2.fill_between(pairs_x-1, (pairs_y - pairs_err), (pairs_y + pairs_err),
                 alpha=0.2)
ax2.set_xlabel('episodes')
ax2.set_ylabel('state / action pairs')
ax2.set_title('State / action pair coverage for DQN')
fig2.savefig('dqn_coverage_3p.png')


with open("win_dqn_data_3p.pkl", "wb") as outfile1:
    pickle.dump(trial_winners, outfile1)
with open("reward_dqn_data_3p.pkl", "wb") as outfile2:
    pickle.dump(trial_rewards, outfile2)
with open("steps_dqn_data_3p.pkl", "wb") as outfile3:
    pickle.dump(trial_steps, outfile3)
with open("pairs_dqn_data_3p.pkl", "wb") as outfile4:
    pickle.dump(trial_pairs, outfile4)
with open("percent_win_data_3p.pkl", "wb") as outfile5:
    pickle.dump(total_percent_win, outfile5)
