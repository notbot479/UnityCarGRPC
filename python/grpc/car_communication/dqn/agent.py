from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (
    MaxPooling2D, 
    Activation, 
    Flatten,
    Dropout, 
    Conv2D, 
    Input,
    Dense, 
)
import tensorflow as tf

from collections import deque
import numpy as np
import random
import time
import os

from config import DQN_LOGS_PATH
from .parameters import *


class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        # Array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        tm = int(time.time())
        log_dir = os.path.join(DQN_LOGS_PATH, f'tm{tm}')
        self.tensorboard = ModifiedTensorBoard(log_dir=log_dir)
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        # TODO create own model
        inputs = Input(shape=(64,64,1))
        outputs = Dense(6, activation='linear')
        model = Sequential([
            inputs,
            Conv2D(32, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Conv2D(32, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Flatten(), 
            Dense(32),
            outputs,
        ])
        model.compile(
            loss="mse", 
            optimizer=Adam(learning_rate=0.001), 
            metrics=['accuracy'],
        )
        return model

    def update_replay_memory(self, transition):
        '''
        Adds step's data to a memory replay array
        (old_state, action, reward, new_state, done)
        '''
        self.replay_memory.append(transition)

    def train(self, terminal_state):
        '''Trains main network every step during episode'''
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE: return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        X, y = [], []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, _, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, 
            # otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        callbacks = [self.tensorboard] if terminal_state else None
        self.model.fit(
            np.array(X),
            np.array(y), 
            batch_size=MINIBATCH_SIZE, 
            verbose=0, 
            shuffle=False, 
            callbacks=callbacks,
        )

        # Update target network counter every episode
        if terminal_state: self.target_update_counter += 1
        # If counter reaches value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        '''
        Queries main network for Q values given current observation space 
        (environment state)
        '''
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        '''
        Overriding init to set initial step and writer 
        (we want one log file for all .fit() calls)
        '''
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model): #pyright: ignore
        '''
        Overriding this method to stop creating default log writer
        '''
        pass

    def on_epoch_end(self, epoch, logs=None): #pyright: ignore
        '''
        Overrided, saves logs with our step number
        (otherwise every .fit() will start writing from 0th step)
        '''
        if logs is not None:
            self.update_stats(**logs)
        else:
            self.update_stats()

    def on_batch_end(self, batch, logs=None): #pyright: ignore
        '''
        Overrided
        We train for one batch only, no need to save anything at epoch end
    
        '''
        pass

    def on_train_end(self, _):
        '''Overrided, so won't close writer'''
        pass
 
    def update_stats(self, **stats):
        '''
        Custom method for saving own metrics
        Creates writer, writes custom metrics and closes writer
        '''
        print(self.step, stats) #TODO write logs

def _test():
    agent = DQNAgent()
    model = agent.create_model()
    model.summary()

if __name__ == '__main__':
    _test()
