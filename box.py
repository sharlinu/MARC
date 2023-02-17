import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import random
import pygame

logging.basicConfig(level=logging.WARNING)
# matplotlib.use("TkAgg")

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


class Entity:
    def __init__(self, x, y, colour):
        self.x = x
        self.y = y
        self.colour = colour


class Object(Entity):
    def __init__(self, x, y, colour, unlocked):
        super().__init__(x, y, colour)
        self.unlocked = unlocked
        self.collected = False


class Player:
    def __init__(self, id):
        self.id = id
        self.position = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.owned_key = 0

    def setup(self, position, field_size):
        self.history = []
        self.position = position
        self.field_size = field_size
        self.score = 0
        self.reward = 0
        self.owned_key = 0

    @property
    def name(self):
        print("Player ", self.id)


class BoxWorldEnv(Env):
    """
    A class that contains rules/actions for the game multi-agent boxworld
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "owned_key", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
            self,
            players,
            field_size,
            num_colours,
            goal_length,
            sight,
            max_episode_steps,
            normalize_reward=True,
            grid_observation=False,
            simple = False,
            single = False,
            relational = False,
            deterministic = False,
    ):
        self.simple = simple
        self.single = single
        self.relational = relational
        self.deterministic = deterministic
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.field = np.zeros(field_size, np.int32)
        self.players = [Player(id=i) for i in range(players)]

        self.num_colours = num_colours
        self.goal_length = goal_length
        if self.simple:
            self.max_objects = 2
        elif self.single:
            self.max_objects = 1
        else:
            self.max_objects = 2 * self.goal_length + 1  # TODO check
        self.sight = sight  # This is to set partial / full observability
        self._game_over = None
        self.gem_colour = 1
        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        # Penalties and rewards
        self.reward_gem = 10
        self.reward_key = 1
        self.reward_unlock = 0
        self.penalty = -1
        self.penalty_overtime = -1

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation  # Boolean: I think the only way to introduce partial observability

        self.action_space = spaces.Tuple(([gym.spaces.Discrete(5)] * len(self.players)))  # TODO simplify?
        if not self._grid_observation:
            self.observation_space = spaces.Tuple(
                ([self._get_observation_space()] * len(self.players)))  # TODO could simplify this to one space?
        else:
            self.single_observation_space = spaces.Dict()
            self.single_observation_space['image'] = self._get_observation_space()
            self.observation_space = spaces.Tuple(
                ([self.single_observation_space] * len(self.players)))
        # self.observation_space = gym.spaces.Tuple(([gym.spaces.Box(low=0, high=self.num_colours, shape=self.field_size, dtype=np.int16)] * len(self.players)))  # TODO could simplify this to one space?

        self.viewer = None  # TODO remove?

        self.n_agents = len(self.players)
        self.n = self.n_agents  # some algorithms call this attribute

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        2-layered grid space where the first layer are the objects and
        2nd layer is the agent position on the board
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_objects = self.max_objects
            max_colour = self.num_colours
            min_obs = [-1, -1, 0] * max_objects + [-1, -1, 0] * len(self.players)
            max_obs = [field_x - 1, field_y - 1, max_colour] * max_objects + [
                field_x - 1,
                field_y - 1,
                max_colour,  # this will show the key that the agent owns
            ] * len(self.players)
        else:
            # grid observation space
            # observation space around center

            max_colour = self.num_colours
            obj_min = np.zeros(self.field_size, dtype=np.int64)
            obj_max = np.ones(self.field_size, dtype=np.int64) * max_colour

            # agents layer: agent levels
            agents_min = np.zeros(self.field_size, dtype=np.int64)
            agents_max = np.ones(self.field_size, dtype=np.int64) * max_colour

            # total layer
            min_obs = np.stack([agents_min, obj_min], axis=2)
            max_obs = np.stack([agents_max, obj_max], axis=2)

        # return gym.spaces.Box(np.array(min_obs), np.array(max_obs), shape=(2, self.field_size[0], self.field_size[1]), dtype=np.int64)
        return gym.spaces.Box(np.array(min_obs), np.array(max_obs))

    @classmethod
    def from_obs(cls, obs):
        """
        Can be called on Env class given an observation
        :param obs:
        :return: ForagingEnv object
        """
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, obs.field.shape)  # TODO why field.shape in setup
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None, None, False)  # constructs a Env class with player parameter
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()
        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }  # returns a dictionary with player and list of possible actions for that player

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        """
        Counts the number of non-zero items that is in surroundings of position
        """
        if not ignore_diag:
            return self.field[
                   max(row - distance, 0): min(row + distance + 1, self.rows),
                   max(col - distance, 0): min(col + distance + 1, self.cols),
                   ]

        return (
                self.field[
                max(row - distance, 0): min(row + distance + 1, self.rows), col
                ].sum()  # returns the sum of the column array in sight
                + self.field[
                  row, max(col - distance, 0): min(col + distance + 1, self.cols)
                  ].sum()  # returns the sum of the row array in sight
            # TODO correct that if you are on a cell with food than it counts double but maybe it never happens
        )

    def adjacent_object(self, row, col):
        """
        Returns adjacent values for food
        :return: value of all non-diagonal adjacent food ASSUMING that (row, col) location is zero! Otherwise double counting of this location
        """
        return (
                self.field[max(row - 1, 0), col]  # on top of of item or item itself if in first row
                + self.field[min(row + 1, self.rows - 1), col]  # below item or item iteself if in last row
                + self.field[row, max(col - 1, 0)]  # left of item
                + self.field[row, min(col + 1, self.cols - 1)]  # right of item
        )

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def sampling_pairs(self, num_pair):
        n = self.field_size[0]
        possibilities = set(range(1, n * (n - 1)))  # n on the y-axis and then (n-1) on the x-axis
        fkeys = []
        keys = []
        locks = []
        for k in range(num_pair + 2):
            key = random.sample(possibilities, 1)[0]
            key_x, key_y = key // (n - 1), key % (n - 1)  # translates possibilities to a grid

            to_remove = [key_x * (n - 1) + key_y] + \
                        [key_x * (n - 1) + i + key_y for i in range(1, min(2, n - 2 - key_y) + 1)] + \
                        [key_x * (n - 1) - i + key_y for i in range(1, min(2, key_y) + 1)]
            # position of the key
            # no objects adjacent or intersecting right
            # no objects adjacently or interesecting left
            possibilities -= set(to_remove)
            if k < 2:
                fkeys.append([key_x, key_y])
                continue
            keys.append([key_x, key_y])
            lock_x, lock_y = key_x, key_y + 1
            locks.append([lock_x, lock_y])
        return fkeys, keys, locks

    def spawn_boxes(self):
        if self.simple:
            self.field[0,0] = 2
            self.objects[(0,0)] = Object(x=0, y=0, colour=2, unlocked=True)
            self.field[3,3] = 2
            self.objects[(3,3)] = Object(x=3, y=3, colour=2, unlocked=True)
        elif self.single:
            self.field[1,1] = 2
            self.objects[(1,1)] = Object(x=1, y=1, colour=2, unlocked=True)
        elif self.relational:
            self.key_colour = np.random.randint(2, self.num_colours)
            # TODO any need for key colour as class object?
            self.spawn_object(self.key_colour, box=True)
            for i in range(self.n_agents):
                self.spawn_object(self.key_colour, box=False)
        elif self.deterministic:
            #self.key_colour = random.randint(2, self.num_colours)
            self.key_colour = 4
            self.spawn_det_object(self.key_colour)
        else:
            self.key_colour = np.random.randint(2, self.num_colours)
            for i in range(self.n_agents):
                self.spawn_object(self.key_colour, box=False)
        return

    def spawn_det_object(self, colour):
        """
        Spawn objects deterministically
        """
        self.field[2,2] = self.gem_colour
        self.gem = Object(2,2, self.gem_colour, unlocked=False)

        self.field[2,1] = colour
        self.objects[(2,1)] = Object(x=2, y=1, colour=colour, unlocked=False)

        self.field[2, 3] = colour
        self.objects[(2,3)] = Object(x=2, y=3, colour=colour, unlocked=False)

        self.field[0, 4] = colour
        self.objects[(0, 4)] = Object(x=0, y=4, colour=colour, unlocked=True)
        self.field[4,4] = colour
        self.objects[(4,4)] = Object(x=4, y=4, colour=colour, unlocked=True)


    def spawn_object(self, colour, box=False):
        """
        Spawns objects in random locations where it is possible!
        """
        attempts = 0
        while attempts < 1000:
            #row = self.np_random.integers(0, self.rows)
            #col = self.np_random.integers(0, self.cols)
            row = np.random.randint(0, self.rows)
            col = np.random.randint(0, self.cols)

            if (self._is_empty_location(row, col)) and (self.adjacent_object(row, col) == 0):
                if box:
                    self.field[row, col] = self.gem_colour
                    self.gem = Object(row, col, self.gem_colour,
                                      unlocked=False)  # TODO check if we ever need that as a class object
                    adj_1, adj_2 = random.sample(self.adjacent_locations(row, col), 2)
                    self.field[adj_1[0], adj_1[1]] = colour
                    self.field[adj_2[0], adj_2[1]] = colour
                    self.objects[(adj_1[0], adj_1[1])] = Object(x=adj_1[0], y=adj_1[1], colour=colour, unlocked=False)
                    self.objects[(adj_2[0], adj_2[1])] = Object(x=adj_2[0], y=adj_2[1], colour=colour, unlocked=False)
                else:
                    self.field[row, col] = colour
                    self.objects[(row, col)] = Object(x=row, y=col, colour=colour, unlocked=True)
                break
            attempts += 1
        return

    # TODO move functions to utils
    def adjacent_locations(self, row, col, complex=True):
        """
        Returns adjacent locations that are within the grid
        :param row: current row
        :param col: current col
        :return: list of adjacent locations
        """
        if complex:
            return [l for l in [(row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)] if
                    self.valid_location(l)]
        else:
            return [l for l in [(row, col - 1), (row, col + 1)] if
                    self.valid_location(l)]

    def valid_location(self, loc):
        """
        Checks if a location is within the grid
        :param loc: (row, col) tuple
        :return: True if location is within grid and False otherwise
        """
        if loc[0] < 0 or loc[0] >= self.rows:
            return False
        elif loc[1] < 0 or loc[1] >= self.cols:
            return False
        else:
            return True

    def spawn_players(self):
        if self.simple:
            self.players[0].setup(
                (2, 2),
                self.field_size,
            )
            self.players[1].setup(
                (1,1),
                self.field_size,
            )
            return
        elif self.single:
            self.players[0].setup(
                (1, 2),
                self.field_size,
            )
            return
        elif self.deterministic:
            self.players[0].setup(
                (0, 0),
                self.field_size,
            )
            self.players[1].setup(
                (4,0),
                self.field_size,
            )
            return
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = np.random.randint(0,self.rows)
                col = np.random.randint(0, self.cols)


                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                    player.position[0] > 0
                # and self.field[player.position[0] - 1, player.position[1]] == 0 # here it test that there is no food on the patch
            )
        elif action == Action.SOUTH:
            return (
                    player.position[0] < self.rows - 1
                # and self.field[player.position[0] + 1, player.position[1]] == 0 # here it test that there is no food on the patch
            )
        elif action == Action.WEST:
            return (
                    player.position[1] > 0
                # and self.field[player.position[0], player.position[1] - 1] == 0 # here it test that there is no food on the patch
            )
        elif action == Action.EAST:
            return (
                    player.position[1] < self.cols - 1
                # and self.field[player.position[0], player.position[1] + 1] == 0 # here it test that there is no food on the patch
            )
        # elif action == Action.LOAD:
        #    return self.adjacent_object(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        # TODO test this - also I think we need to transform location of food as well!
        """
        Transforms neighbourhood
        :param center: location of reference perspecitve
        :param sight: (sight,sight) is the field the agent can see around the center
        :param position: location of another player
        :return: Adjusted location given sight and player's perspective
        """
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        """
        Makes observation from agent "player" perspective
        :param player: The agent from which we base the observation
        :return:
        """
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    owned_key=a.owned_key,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                           min(
                               self._transform_to_neighborhood(
                                   player.position, self.sight, a.position
                               )
                           )
                           >= 0
                   )
                   and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                   <= 2 * self.sight
            ],
            # TODO also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),  # TODO check what this gives me?
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            """
            Adds position and values for food and players if observation is non-grid
            :param observation: Named tuple with attributes
            :return: gymobservation-space-compatible array
            """

            obs = np.zeros(self.observation_space[0].shape, dtype=np.int64)
            # assert self.observation_space[0].shape[0] == (3*self.max_objects + 3*self.n_agents)
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_objects):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]  # TODO what is this?

            for i in range(len(self.players)):
                obs[self.max_objects * 3 + 3 * i] = -1
                obs[self.max_objects * 3 + 3 * i + 1] = -1
                obs[self.max_objects * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_objects * 3 + 3 * i] = p.position[0]
                obs[self.max_objects * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_objects * 3 + 3 * i + 2] = p.owned_key

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays if grid observation space
            """
            grid_shape = self.field_size
            agents_layer = np.zeros(grid_shape, dtype=np.int64)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x, player_y] = 1

            foods_layer = self.field.copy()

            # agent locations are not accessible
            # food locations are not accessible
            return np.stack([agents_layer, foods_layer], axis=2)


        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]  # Returns a named tuple
        if self._grid_observation:
            layers = make_global_grid_arrays()
            nobs = {'image': layers}
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        ninfo = {}

        # check the space of obs
        # for i, obs in enumerate(nobs):
        #    assert self.observation_space[i].contains(obs), \
        #        "observation space mismatch"
        # f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"

        return nobs, np.array(nreward), np.array(ndone), ninfo

    def reset(self):
        # logging.warning('environment is reset')
        self.field = np.zeros(self.field_size, np.int32)
        self.objects = {}
        self.spawn_players()
        # player_levels = sorted([player.level for player in self.players])

        self.spawn_boxes()

        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs, _, _, _ = self._make_gym_obs()
        return nobs

    def step(self, actions: object) -> object:
        self.current_step += 1
        cnt = 0
        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.warning(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)  # keys will be position and value the list of players on the position

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)

        # do movements for non colliding players or finish game if they reach the gem
        for k, v in collisions.items():
            if len(v) > 1:  # value lists longer than 1 mean colliding players
                continue
#                if self.field[k] == 1:
#                    if self.gem.unlocked == True:
#                        for p in v:
#                            logging.warning('This should currently not happen in easier setting')
#                            p.reward = self.reward_gem
#                        self._game_over = True  # TODO can be changed at the end as an add condition self.gem.collected=True
#                else:
#                    self.logger.info("colliding location is not the unlocked gem")
#                    continue

            # this goes through all new positions of non-colliding players (len(v)==1) and checks that you cannot go on

            # agent cannot move on lock without having the key
            elif (k in self.objects.keys()
                  and self.objects[k].unlocked == False
                  and self.objects[k].colour != v[0].owned_key):
                # invalid action because agent does not have the key for lock it wants to move on
                self.logger.info("Agent cannot move onto locked objects")
                continue
            try:
                if (k == (self.gem.x, self.gem.y) and self.gem.unlocked == False):
                    # cannot move onto locked gem
                    self.logger.info("Agent cannot move onto locked gem")
                    continue
            except AttributeError:
                pass
            # position needs to be on the grid
            assert 0 <= (k[0] and k[1]) < self.field_size[0]
            v[0].position = k  # now k is the position of non-colliding player
            if v[0].history:
                # no diagonal moves allowed
                assert abs(v[0].history[-1][0] - k[0]) + abs(v[0].history[-1][1] - k[1]) <= 1
            v[0].history.append(k)
        # pick up key if they walk on one.
        for player in self.players:
            # player wants to walk on an object
            if self.field[player.position] != 0:
                self.logger.info("agent steps on an object")
                # check if it is a key
                if (player.position in self.objects.keys()
                        and self.objects[player.position].unlocked == True
                        and self.objects[player.position].collected == False):
                    self.objects[player.position].collected = True
                    if player.owned_key != 0:
                        logging.warning('player has a key and receives penalty')
                        player.reward = self.penalty
                        self._game_over = True
                    else:
                        player.owned_key = self.objects[player.position].colour
                        # after pick up the key disappears from the grid
                        self.field[player.position] = 0
                        player.reward = self.reward_key
                        print('player picks up key')

                # check if agent moves on a lock
                elif (player.position in self.objects.keys()
                      and self.objects[player.position].unlocked == False
                      and self.objects[player.position].collected == False
                      and self.objects[player.position].colour == player.owned_key):
                    cnt += 1
        #            else:
        #                # penalty for every step the agent does not reach the goal
        #                p.reward = self.penalty
        # check that all players are on locks
        if cnt == len(self.players):
            logging.warning('This is exactly where agents should be and finish')
            for player in self.players:
                # both locks need to be switched to collected
                self.objects[player.position].collected = True
                self.objects[player.position].unlocked = True
                # both need to be uncoloured / removed
                self.field[player.position] = 0
                # gem needs to be unlocked
                self.gem.unlocked = True
                # player.reward = self.reward_unlock
                player.reward = self.reward_gem # TODO currently automatic collection
            self._game_over = True

        if self._game_over is False:
            self._game_over = (
                    self.field.sum() == 0
                    or self._max_episode_steps <= self.current_step
            )

        if self._max_episode_steps <= self.current_step:
            for p in self.players:
                p.reward = self.penalty_overtime
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        return self._make_gym_obs()

    def _init_render(self):
        from r_maac.box_rendering import Viewer
        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, return_rgb_array=True)

    def close(self):
        if self.viewer:
            self.viewer.close()


def _game_loop(env, render=False):
    """
    """
    env.reset()
    done = False

    if render:
        env.render()
        # pygame.display.update()  # update window
        time.sleep(1)

    while not done:
        # for i in range(100):
        actions = env.action_space.sample()

        nobs, nreward, ndone, _ = env.step(actions)
        # nobs, nreward, ndone, _ = env.step((1,1))
        if sum(nreward) != 0:
            print(nreward)
            print(nobs)

        time.sleep(2)

        if render:
            env.render()
            # pygame.display.update()  # update window
            time.sleep(0.5)

        done = np.all(ndone)
        pygame.event.pump()  # process event queue
    # print(env.players[0].score, env.players[1].score)


if __name__ == "__main__":
    env = BoxWorldEnv(
        players=1,
        field_size=(4,4),
        num_colours=5,
        goal_length=1,
        sight=4,
        max_episode_steps=500,
        grid_observation= True,
        simple=False,
        single = True,
        relational = False,
        deterministic= True,
    )
    background_colour = (50, 50, 50)
    for episode in range(1):
        _game_loop(env, render=True)
    # nobs, nreward, ndone, ninfo = env.step([1,1])
    print("Done")

# TODO add owned key information visually somewhere?
# TODO sometimes agent disappears from visual

# integrate MAAC and see if it works
# MADDPG alone
# MADDPG + R-GCN
# Arxiv mailing list
# temporal graphs etc?
