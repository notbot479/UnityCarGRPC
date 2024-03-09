from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Model
from keras.saving import load_model
from keras.layers import (
    BatchNormalization,
    MaxPooling2D,
    Concatenate,
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


def extract_inputs(data: list) -> list[np.ndarray]:
    return [np.array(column) for column in zip(*data)]

class DQNAgent:
    def __init__(self, *, filepath:str | None = None):
        # create main and target model
        self.model = load_model(filepath) if filepath else self.create_model()
        self.target_model = self.create_model()
        weights = self.model.get_weights() #pyright: ignore
        self.target_model.set_weights(weights)
        # Array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Custom tensorboard object
        tm = int(time.time())
        log_dir = os.path.join(DQN_LOGS_PATH, f'tm{tm}')
        self.tensorboard = ModifiedTensorBoard(log_dir=log_dir)
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        # define model inputs and outputs
        image = Input(shape=(64,64,1), name='image_input')
        distance_sensors_distances = Input(
            shape=(6,),
            name='distance_sensors_distances_input',
        )
        distance_to_target_router = Input(
            shape=(1,),
            name='distance_to_target_router_input',
        )
        in_target_area = Input(shape=(1,), name='in_target_area_input')
        boxes_is_found = Input(shape=(1,), name='boxes_is_found_input')
        distance_to_box = Input(shape=(1,), name='distance_to_box_input')
        target_found = Input(shape=(1,), name='target_found_input') 
        inputs = [
            image,
            distance_sensors_distances,
            distance_to_target_router,
            in_target_area,
            boxes_is_found,
            distance_to_box,
            target_found,
        ]
        
        # create model layers
        x_image = Conv2D(16, (3, 3), padding='same')(image)
        x_image = BatchNormalization()(x_image)
        x_image = Activation('relu')(x_image)
        x_image = MaxPooling2D(pool_size=(2, 2))(x_image)
        x_image = Dropout(0.2)(x_image)
        x_image = Conv2D(32, (3, 3), padding='same')(x_image)
        x_image = BatchNormalization()(x_image)
        x_image = Activation('relu')(x_image)
        x_image = MaxPooling2D(pool_size=(2, 2))(x_image)
        x_image = Dropout(0.2)(x_image)
        x_image = Flatten()(x_image)

        x_sensors = Dense(6, activation='relu')(distance_sensors_distances)
        x_router = Dense(1, activation='relu')(distance_to_target_router)
        x_in_target_area = Dense(1,activation='relu')(in_target_area)
        x_boxes_is_found = Dense(1, activation='relu')(boxes_is_found)
        x_distance_to_box = Dense(1, activation='relu')(distance_to_box)
        x_target_found = Dense(1, activation='relu')(target_found)

        concatenated = Concatenate()([
            x_image,
            x_sensors,
            x_router,
            x_in_target_area,
            x_boxes_is_found,
            x_distance_to_box, 
            x_target_found,
        ])

        x = Dense(128, activation='relu')(concatenated)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(6, activation='linear')(x)
        
        # create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss="mse", 
            metrics=['accuracy'],
        )
        return model

    def update_replay_memory(self, transition):
        '''
        Adds step's data to a memory replay array
        (old_state, action, reward, new_state, done)
        '''
        self.replay_memory.append(transition)

    @property
    def train_available(self) -> bool:
        '''Start training only if certain number of samples is already saved'''
        return len(self.replay_memory) > MIN_REPLAY_MEMORY_SIZE

    def update_target_network_weights(self) -> None:
        weights = self.model.get_weights() #pyright: ignore
        self.target_model.set_weights(weights)
        self.target_update_counter = 0

    def train_on_episode_end(self, *, batches_count: int = 50) -> None:
        if not(self.train_available): return
        rng = range(batches_count)
        batches = [random.sample(self.replay_memory, MINIBATCH_SIZE) for _ in rng]
        for minibatch in batches: 
            self._train(minibatch=minibatch)
        # Update target network counter
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.update_target_network_weights()

    def train(self, terminal_state: bool) -> None:
        '''Trains main network every step during episode'''
        if not(self.train_available): return
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE: return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        self._train(minibatch=minibatch)
        # Update target network counter every episode
        if terminal_state: 
            self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.update_target_network_weights()
    
    def _train(self, minibatch) -> None:
        # show stats
        i = self.target_update_counter + 1
        print(f'[{i}] Train model. Minibatch size: {len(minibatch)}')
        # Get current states from minibatch, then query NN model for Q values
        current_states_X = extract_inputs(
            [transition[0] for transition in minibatch],
        )
        current_qs_list = self.model.predict( #pyright: ignore
            current_states_X, 
            verbose=0,
        ) 
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should queried
        new_current_states_X = extract_inputs(
            [transition[3] for transition in minibatch],
        )
        future_qs_list = self.target_model.predict(
            new_current_states_X, 
            verbose=0,
        )
        # Now we need to enumerate our batches
        X, y = [], []
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
        #callbacks = [self.tensorboard] if terminal_state else None
        X_extracted = extract_inputs(X)
        self.model.fit( #pyright: ignore
            X_extracted,
            np.array(y), 
            batch_size=MINIBATCH_SIZE, 
            verbose=0, 
            shuffle=False, 
            #callbacks=callbacks,
        )

    def get_qs(self, state):
        '''
        Queries main network for Q values given current observation space 
        (environment state)
        '''
        X = extract_inputs([state,])
        y = self.model.predict(X, verbose=0)[0] #pyright: ignore
        return y

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
        print(self.step, stats) #TODO write logs [LogWriter]

def _test():
    agent = DQNAgent()
    model = agent.create_model()
    model.summary()

if __name__ == '__main__':
    _test()
