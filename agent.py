import os
import random
import collections
import numpy as np
import tensorflow as tf
from tensorboard import TensorflowLogger
from keras import models, layers, optimizers

class UnoAgent:

    def __init__(self, num_possible_moves, possible_actions):
        self.initialized = False
        self.logger = TensorflowLogger('logs')
        self.model = self.build_model(num_possible_moves, possible_actions)
        self.model_final = self.build_model(num_possible_moves, possible_actions)
        self.model_final.set_weights(self.model.get_weights())
        self.replay_memory = collections.deque(maxlen=10000)

    def build_model(self, _input, _output):
        model = models.Sequential()
        model.add(layers.Dense(units=128, activation='relu', input_shape=(_input,)))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=_output, activation='linear'))
        model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def predict(self, state):
        return np.argmax(self.model.predict(np.array(state).reshape(-1, *state.shape))[0])

    def train(self):
        i = 0
        while True:
            if len(self.replay_memory) < 512:
                continue
            
            batch = np.array(random.sample(self.replay_memory, 512))
            states = np.array(list(batch[:,0]))
            q_values = self.model.predict(states)
            max_future_q = np.max(self.model.predict(np.array(list(minibatch[:,3]))), axis=1)

            for j in range(len(batch)):
                action, reward, done = batch[j,1], batch[j,2], batch[j,4]
                q_values[j,action] = reward
                if not done:
                    q_values[j,action] += 0.9 * max_future_q[j]

            history = self.model_final.fit(x=states, y=q_values, 512, verbose=0)
            self.logger.scalar('loss', history.history['loss'][0])
            self.logger.scalar('accuracy', history.history['accuracy'][0])
            self.logger.flush()
            
            i += 1
            if i % 50 == 0:
                self.model.set_weights(self.model_final.get_weights())
                if not self.initialized:
                    self.initialized = True
            
            if i % 1000 == 0:
                self.model.save(f'models/{self.logger.timestamp}'/model-{i}.h5')
