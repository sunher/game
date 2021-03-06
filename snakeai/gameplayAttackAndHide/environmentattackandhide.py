import pprint
import random
import time

import numpy as np
import pandas as pd

from .entities import Snake, Field, CellType, SnakeAction, ALL_SNAKE_ACTIONS, SnakeDirection


class EnvironmentAttackAndHide(object):
    """
    Represents the RL environment for the Snake game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def __init__(self, config, verbose=1):
        """
        Create a new Snake RL environment.

        Args:
            config (dict): level configuration, typically found in JSON configs.
            verbose (int): verbosity level:
                0 = do not write any debug information;
                1 = write a CSV file containing the statistics for every episode;
                2 = same as 1, but also write a full log file containing the state of each timestep.
        """
        self.field = Field(level_map=config['field'])
        self.snake = None
        self.fruit = []
        self.poison = []
        self.poison_num = 0
        self.initial_snake_length = config['initial_snake_length']
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 1000)
        self.is_game_over = False

        self.timestep_index = 0
        self.current_action = None
        self.stats = EpisodeStatistics()
        self.verbose = verbose
        self.debug_file = None
        self.stats_file = None
        self.enemy = None

    def seed(self, value):
        """ Initialize the random state of the environment to make results reproducible. """
        random.seed(value)
        np.random.seed(value)

    @property
    def observation_shape(self):
        """ Get the shape of the state observed at each timestep. """
        return self.field.size, self.field.size

    @property
    def num_actions(self):
        """ Get the number of actions the agent can take. """
        return len(ALL_SNAKE_ACTIONS)

    def new_episode(self):
        """ Reset the environment and begin a new episode. """
        self.field.create_level()
        self.stats.reset()
        self.timestep_index = 0

        self.enemy = None
        self.fruit = []
        self.poison = []
        self.poison_num = 0
        self.snake = Snake(self.field.get_random_empty_cell(), length=self.initial_snake_length)
        self.field.place_snake(self.snake)
        self.generate_emeny()
        self.generate_poison()
        self.current_action = None
        self.is_game_over = False

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def record_timestep_stats(self, result):
        """ Record environment statistics according to the verbosity level. """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Write CSV header for the stats file.
        if self.verbose >= 1 and self.stats_file is None:
            self.stats_file = open('snake-env-{timestamp}.csv', 'w')
            stats_csv_header_line = self.stats.to_dataframe()[:0].to_csv(index=None)
            # print(stats_csv_header_line, self.stats_file, '', flush=True)

        # # Create a blank debug log file.
        # if self.verbose >= 2 and self.debug_file is None:
        #     self.debug_file = open('snake-env-{timestamp}.log', 'w')

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        # if self.verbose >= 2:
        #     print(result, self.debug_file)
        #
        # # Log episode stats if the appropriate verbosity level is set.
        # if result.is_episode_end:
        #     if self.verbose >= 1:
        #         stats_csv_line = self.stats.to_dataframe().to_csv(header=False, index=None)
        #         print(stats_csv_line, self.stats_file, '', flush=True)
        #     if self.verbose >= 2:
        #         print(self.stats, self.debug_file)

    def get_observation(self):
        """ Observe the state of the environment. """
        return np.copy(self.field._cells)

    def choose_action(self, action):
        """ Choose the action that will be taken at the next timestep. """

        self.current_action = action
        if action == SnakeAction.TURN_LEFT1:
            self.snake.turn_left()
        elif action == SnakeAction.TURN_LEFT2:
            self.snake.turn_left()
            self.snake.turn_left()
        elif action == SnakeAction.TURN_LEFT3:
            self.snake.turn_left()
            self.snake.turn_left()
            self.snake.turn_left()
        elif action == SnakeAction.TURN_RIGHT1:
            self.snake.turn_right()
        elif action == SnakeAction.TURN_RIGHT2:
            self.snake.turn_right()
            self.snake.turn_right()
        elif action == SnakeAction.TURN_RIGHT3:
            self.snake.turn_right()
            self.snake.turn_right()
            self.snake.turn_right()

    def generate_emeny(self, position=None):
        """ Generate a new fruit at a random unoccupied cell. """
        if position is None:
            position = self.field.get_random_empty_cell()
            self.enemy = position
        self.field[position] = CellType.SNAKE_BODY
        if np.random.random() > 0.2:
            if (self.field[position + SnakeDirection.NORTH] == CellType.EMPTY):
                self.field[position + SnakeDirection.NORTH] = CellType.FRUIT
                self.fruit.append(position + SnakeDirection.NORTH)
            if (self.field[position + SnakeDirection.SOUTH] == CellType.EMPTY):
                self.field[position + SnakeDirection.SOUTH] = CellType.FRUIT
                self.fruit.append(position + SnakeDirection.SOUTH)
            if (self.field[position + SnakeDirection.WEST] == CellType.EMPTY):
                self.field[position + SnakeDirection.WEST] = CellType.FRUIT
                self.fruit.append(position + SnakeDirection.WEST)
            if (self.field[position + SnakeDirection.EAST] == CellType.EMPTY):
                self.field[position + SnakeDirection.EAST] = CellType.FRUIT
                self.fruit.append(position + SnakeDirection.EAST)
            if np.random.random() < 0.1:
                position = self.field.get_random_empty_cell()
                self.field[position] = CellType.FRUIT
                self.fruit.append(position)
            if np.random.random() < 0.1:
                position = self.field.get_random_empty_cell()
                self.field[position] = CellType.FRUIT
                self.fruit.append(position)

    def generate_poison(self):
        """ Generate a new fruit at a random unoccupied cell. """
        if np.random.random() < 0:
            self.poison_num = random.Random().choice([2, 3])
            for position in self.field.get_empty_cell():
                if (0 < position.x <= self.poison_num or 0 < position.y <= self.poison_num or (
                        position.x + self.poison_num) >= (self.field.size - 1) or (position.y + self.poison_num) >= (
                        self.field.size - 1)):
                    self.field[position] = CellType.POISON
                    self.poison.append(position)

    def be_poison(self, position):
        """ Generate a new fruit at a random unoccupied cell. """
        # if np.random.random() < 1:
        if (0 < position.x <= self.poison_num or 0 < position.y <= self.poison_num or (
                position.x + self.poison_num) >= (self.field.size - 1) or (position.y + self.poison_num) >= (
                self.field.size - 1)):
            return True
        return False

    def timestep(self):
        """ Execute the timestep and return the new observable state. """

        self.timestep_index += 1
        reward = 0
        isdie = False
        old_head = self.snake.head
        old_tail = self.snake.tail

        # Are we about to eat the fruit?
        if self.fruit.__contains__(self.snake.peek_next_move()):
            self.fruit.remove(self.snake.peek_next_move())
            # self.generate_fruit()
            # old_tail = None
            reward += self.rewards['ate_fruit']
            self.stats.fruits_eaten += 1
        elif self.be_poison(self.snake.peek_next_move()):
            self.stats.poisons_eaten += 1
        # If not, just move forward.

        self.snake.move()

        self.field.update_snake_footprint(old_head, old_tail, self.snake.head)

        # Hit a wall or own body?
        if not self.is_alive():
        #    reward -=self.fruit.__len__()
            if self.has_hit_wall() or self.has_hit_own_body():
                self.stats.termination_reason = 'hit_wall'
                reward -= 0.7
                isdie = True
            self.field[self.snake.head] = CellType.SNAKE_HEAD
            self.is_game_over = True
            # reward *= 0.7
            # print(self.fruit.__len__())
            # if(self.get_wall_num(old_head) >= 2) and self.fruit.__len__()<=1:
            #     reward = self.get_wall_num(old_head) - self.fruit.__len__()
            # else:
            #     reward = -1
            reward += (self.get_wall_num(old_head)-1.5)
            if self.snake.length == 2 or self.snake.length == 1:
                reward -= 2

            if self.stats.poisons_eaten != 0:
                reward -= 2

            if (self.be_poison(old_head)):
                reward -= 1

            # reward += 0.99
        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result

    def get_wall_num(self, position=None):
        num = 0
        if self.field[position + SnakeDirection.NORTH] == CellType.WALL:
            num += 1
        if self.field[position + SnakeDirection.SOUTH] == CellType.WALL:
            num += 1
        if self.field[position + SnakeDirection.WEST] == CellType.WALL:
            num += 1
        if self.field[position + SnakeDirection.EAST] == CellType.WALL:
            num += 1
        if self.field[
            position + SnakeDirection.NORTH] == CellType.POISON:
            num += 0.5
        if self.field[
            position + SnakeDirection.SOUTH] == CellType.POISON:
            num += 0.5
        if self.field[
            position + SnakeDirection.WEST] == CellType.POISON:
            num += 0.5
        if self.field[
            position + SnakeDirection.EAST] == CellType.POISON:
            num += 0.5
        return num

    def generate_fruit(self, position=None):
        """ Generate a new fruit at a random unoccupied cell. """
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FRUIT
        self.fruit = position

    def has_hit_wall(self):
        """ True if the snake has hit a wall, False otherwise. """
        return self.field[self.snake.head] == CellType.WALL

    def has_hit_own_body(self):
        """ True if the snake has hit its own body, False otherwise. """
        return self.field[self.snake.head] == CellType.SNAKE_BODY

    def is_alive(self):
        """ True if the snake is still alive, False otherwise. """
        return not self.has_hit_wall() and not self.has_hit_own_body()


class TimestepResult(object):
    """ Represents the information provided to the agent after each timestep. """

    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return '{field_map}\nR = {self.reward}   {self.is_episode_end}\n'


class EpisodeStatistics(object):
    """ Represents the summary of the agent's performance during the episode. """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Forget all previous statistics and prepare for a new episode. """
        self.timesteps_survived = 0
        self.sum_episode_rewards = 0
        self.fruits_eaten = 0
        self.poisons_eaten = 0
        self.termination_reason = None
        self.action_counter = {
            action: 0
            for action in ALL_SNAKE_ACTIONS
        }

    def record_timestep(self, action, result):
        """ Update the stats based on the current timestep results. """
        self.sum_episode_rewards += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self):
        """ Format all episode statistics as a flat object. """
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards': self.sum_episode_rewards,
            'mean_reward': self.sum_episode_rewards / self.timesteps_survived if self.timesteps_survived else None,
            'fruits_eaten': self.fruits_eaten,
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_SNAKE_ACTIONS
        })
        return flat_stats

    def to_dataframe(self):
        """ Convert the episode statistics to a Pandas data frame. """
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())
