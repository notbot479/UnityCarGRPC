from tensorflow.keras import layers, models #pyright: ignore
import tensorflow as tf
from config import *


def create_dqn_model():
    shape = tuple(list(VIDEO_FRAME_SIZE)+[3])
    input_image = layers.Input(shape=shape, name='video_frame')
    input_sensors = layers.Input(shape=(DISTANCE_SENSORS_COUNT,), name='sensors_data')
    input_collision = layers.Input(shape=(1,), name='car_collide_obstacle')

    x = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_image)
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    y = layers.Dense(32, activation='relu')(input_sensors)

    combined = layers.concatenate([x, y])
    combined = layers.concatenate([combined, input_collision])
    total = layers.Dense(512, activation='relu')(combined)
    
    inputs = [input_image,input_sensors,input_collision]
    output = layers.Dense(
        MODEL_OUTPUT_COMMANDS_COUNT, 
        activation='softmax', 
        name='output',
        )(total)
    model = models.Model(
        inputs=inputs, 
        outputs=output,
        )
    model.compile(optimizer='adam', loss='mse')
    return model

_dqn_model = create_dqn_model()

def _test():
    _dqn_model.summary()

if __name__ == '__main__':
    _test()