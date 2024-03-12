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
import re
import os

from config import DQN_LOGS_PATH, DQN_MODELS_PATH
from .parameters import *

#model_min_reward[-1.54]_max_reward[28.44]_average_reward[13.347]_1710233149.7119274.keras
def parse_metrics_from_model_name(model_name:str) -> dict:
    model_name = model_name[5::] # remove `model` from name
    pattern = r"(\w+)\[(-?\d+\.\d+)\]"
    matches = re.findall(pattern, model_name)
    metrics = {}
    for match in matches:
        metric_name = match[0][1::]
        metric_value = float(match[1])
        metrics[metric_name] = metric_value
    return metrics

def get_best_model_path(*, models_path: str | None = None, metric:str = 'average_reward') -> str:
    models_path =  models_path if models_path else DQN_MODELS_PATH
    data = []
    for model_name in os.listdir(models_path):
        model_path = os.path.join(models_path, model_name)
        metrics = parse_metrics_from_model_name(model_name)
        m = metrics.get(metric)
        if not(m): continue 
        data.append((model_path, m))
    if not(data): return ''
    best = max(data, key=lambda d: d[1])
    return best[0]

def extract_inputs(data: list) -> list[np.ndarray]:
    return [np.array(column) for column in zip(*data)]

class DQNAgent:
    def __init__(self, *, filepath:str | None = None):
        # create main and target model
        if filepath:
            self.model = load_model(filepath) 
            print(f'Model was loaded: {filepath}')
        else:
            self.model = self.create_model()
        self.target_model = self.create_model()
        weights = self.model.get_weights() #pyright: ignore
        self.target_model.set_weights(weights)
        # Array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Custom tensorboard object
        tm = int(time.time())
        log_dir = os.path.join(DQN_LOGS_PATH, f'{tm}')
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

        x = Dense(512, activation='relu')(concatenated)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(5, activation='linear')(x)
        
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
    def replay_memory_sample(self) -> list:
        '''get random sample from reply memory fixed length (based - dqn parameters)'''
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        return minibatch

    @property
    def train_available(self) -> bool:
        '''Start training only if certain number of samples is already saved'''
        return len(self.replay_memory) > MIN_REPLAY_MEMORY_SIZE

    def update_target_network_weights(self) -> None:
        weights = self.model.get_weights() #pyright: ignore
        self.target_model.set_weights(weights)
        self.target_update_counter = 1 # set 1 instead 0 after update weights

    def train_on_episode_end(self, *, batches_count: int = 50) -> None:
        if not(self.train_available): return
        # Update target network counter
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.update_target_network_weights()
        # generate batch (minibatches * batches_count)
        batch = []
        [batch.extend(self.replay_memory_sample) for _ in range(batches_count)]
        if not(batch): return
        # train on batch and update tensorboard (set terminal state True)
        self._train(minibatch=batch, terminal_state=True)
    
    def train(self, terminal_state: bool) -> None:
        '''Trains main network every step during episode'''
        if not(self.train_available): return
        if terminal_state: self.target_update_counter += 1
        # Update target network counter every episode
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.update_target_network_weights()
        # Get a minibatch of random samples from memory replay table
        minibatch = self.replay_memory_sample
        self._train(minibatch=minibatch, terminal_state=terminal_state)
    
    def _train(
        self, 
        minibatch:list, 
        *, 
        terminal_state:bool = False,
        batch_size: int | None = None,
        ) -> None:
        if not(batch_size): batch_size = MINIBATCH_SIZE
        # show some stats
        i = self.target_update_counter
        b = len(minibatch)//batch_size
        print(f'[{i}] Train model. Minibatch size: {batch_size}. Batches: {b}')
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
        callbacks = [self.tensorboard,] if terminal_state else None
        X_extracted = extract_inputs(X)
        self.model.fit( #pyright: ignore
            X_extracted,
            np.array(y), 
            batch_size=batch_size,
            verbose=0, 
            shuffle=False, 
            callbacks=callbacks,
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
    def __init__(self, log_dir:str, **kwargs):
        '''
        Overriding init to set initial step and writer 
        (we want one log file for all .fit() calls)
        '''
        super().__init__(log_dir=log_dir, **kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(log_dir)

    def set_model(self, model): #pyright: ignore
        '''
        Overriding this method to stop creating default log writer
        '''
        self._model = model
        self._log_write_dir = self.log_dir

        self._train_dir = os.path.join(self._log_write_dir, "train")
        self._train_step = 0

        self._val_dir = os.path.join(self._log_write_dir, "validation")
        self._val_step = 0
        
        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None): #pyright: ignore
        '''
        Overrided, saves logs with our step number
        (otherwise every .fit() will start writing from 0th step)
        '''
        logs = logs if logs else {}
        self.update_stats(**logs)

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
        print(f'\nTensorBoard write logs. Step: {self.step}. Stats: {stats}\n') 
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()


def _test():
    agent = DQNAgent()
    model = agent.create_model()
    model.summary()

if __name__ == '__main__':
    _test()
