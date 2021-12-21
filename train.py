import sys
import threading
import numpy as np
from agent import UnoAgent
from environment import UnoEnvironment


def train(agent):
    epsilon = 1
    env = UnoEnvironment(4)

    while True:
        done = False
        state = None
        rewards = []
        while not done:
            if state is None or np.random.sample() < epsilon or not agent.initialized:
                action = np.random.randint(env.possible_actions())
            else:
                action = agent.predict(state)

            _state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if state is not None:
                # include the current transition in the replay memory
                agent.update_replay_memory((state, action, reward, _state, done))
            state = _state

            if agent.initialized:
                epsilon *= 0.9999
                epsilon = max(epsilon, 0.01)
                
        agent.logger.scalar('cumulative_reward', np.sum(rewards))
        agent.logger.scalar('mean_reward', np.mean(rewards))
        agent.logger.scalar('game_length', len(rewards))
        agent.logger.scalar('epsilon', epsilon)

        env.reset()

env = UnoEnvironment(1)
agent = UnoAgent(env.num_possible_moves(), dummy_env.possible_actions(), _path='models/model-10000.h5')

for _ in range(4):
    threading.Thread(target=train, args=(agent,), daemon=True).start()
    agent.train()
