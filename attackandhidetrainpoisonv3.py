#!/usr/bin/env python3.6

""" Front-end script for training a Snake agent. """

import json
import sys

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from snakeai.agent.dqnxx import DoubleDeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.gameplay4eat.environment4eat import Environment4Eat
from snakeai.gameplayAttackAndHide.environmentattackandhide import EnvironmentAttackAndHide
from snakeai.gameplayAttackAndHidePoison.environmentattackandhidepoison import EnvironmentAttackAndHidePoison
from snakeai.gameplayAttackAndHideRandom.environmentattackandhiderandom import EnvironmentAttackAndHideRandom
from snakeai.gameplayAttackAndHideRandomPoison.environmentattackandhiderandompoison import \
    EnvironmentAttackAndHideRandomPoison
from snakeai.utils.cli import HelpOnFailArgumentParser


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level1',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--level2',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--level3',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--level4',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--level5',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--level6',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000,
        help='The number of episodes to run consecutively.',
    )

    # parser.add_argument(
    #     '--model',
    #     required=True,
    #     type=str,
    #     help='JSON file containing a level definition.',
    # )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return EnvironmentAttackAndHidePoison(config=env_config, verbose=1)


def create_snake_environment4eat(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return EnvironmentAttackAndHideRandomPoison(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.

    Args:
        env: an instance of Snake environment.
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """

    inp = Input(shape=((num_last_frames,) + env.observation_shape))
    x = Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_first')(inp)
    x = Activation('relu')(x)
    x = Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, kernel_size=(2, 2), strides=(1, 1), data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, kernel_size=(2, 2), strides=(1, 1), data_format='channels_first')(x)
    x = Activation('relu')(x)

    layer_shared2 = Flatten()(x)
    print("Shared layers initialized....")

    x = Dense(1024, activation='relu', kernel_initializer='he_uniform')(layer_shared2)
    x = Activation('relu')(x)
    x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    h = Activation('relu')(x)
    y = Dense(env.num_actions + 1)(h)
    z = Lambda(lambda a: K.expand_dims(a[:, 0]) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
               output_shape=(env.num_actions,))(y)
    # layer_q = Lambda(lambda x: x[0][:] + x[1][:] - K.mean(x[1][:]), output_shape=(env.action_space.n,))([layer_v, layer_a])
    # layer_q = Lambda(lambda x: x[0][:] + x[1][:] - K.mean(x[1][:]), output_shape=(env.num_actions,))(y)

    print("Q-function layer initialized.... :)\n")

    model = Model(inp, z)
    model.summary()
    model.compile(optimizer=Adam(), loss='mse')
    return model


def load_model(filename):
    """ Load a pre-trained agent model. """

    from keras.models import load_model
    return load_model(filename)


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    env1 = create_snake_environment(parsed_args.level1)
    env2 = create_snake_environment(parsed_args.level2)
    env3 = create_snake_environment(parsed_args.level3)
    env4 = create_snake_environment(parsed_args.level4)
    env5 = create_snake_environment(parsed_args.level5)
    env6 = create_snake_environment4eat(parsed_args.level6)
    # model = load_model(parsed_args.model) if parsed_args.model is not None else None
    # target_model = load_model(parsed_args.model) if parsed_args.model is not None else None
    model = create_dqn_model(env1, num_last_frames=4)
    target_model = create_dqn_model(env1, num_last_frames=4)
    agent = DoubleDeepQNetworkAgent(
        model=model,
        target_model=target_model,
        memory_size=-1,
        num_last_frames=model.input_shape[1]
    )
    agent.train(
        env1,
        env1,
        env1,
        env1,
        env1,
        env1,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10, exploration_range=(1.0, 0.1),
        discount_factor=0.95
    )


if __name__ == '__main__':
    main()
