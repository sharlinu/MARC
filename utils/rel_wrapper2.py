from gym_minigrid.minigrid import DIR_TO_VEC  # DIR vec is [array([1, 0]), array([0, 1]), array([-1,  0]), array([ 0, -1])]
import numpy as np
import gym
import time
import pygame
import torch
from torch_geometric.data import Data as GeometricData
class GridObject:
    "object is specified by its location"
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.attributes = attributes

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def name(self):
        return "_"+str(self.x)+str(self.y)

# from random import randint
class AbsoluteVKBWrapper(gym.ObservationWrapper):
    """
    Add a vkb key-value pair, which represents the state as a vectorised knowledge base.
    Entities are objects in the gird-world, predicates represents the properties of them and relations between them.
    """
    def __init__(self, env, dense, background_id="b3"):
        super().__init__(env, new_step_api=True)
        self.attribute_labels = ['agents', 'id', 'feature']
        self.n_attr = len(self.attribute_labels)
        self.dense = dense
        self.background_id = background_id
        self.rgcn = False
        self.rel_deter_func = self.id_to_rule_list(self.background_id)
        self.n_rel_rules = len(self.rel_deter_func)
        # number of objects/entities are the number of cells on the grid
        if not self.dense:
            self.obj_n = np.prod(env.observation_space[0]['image'].shape[:-1]) #physical entities
        else:
            self.obj_n = env.n_agents + env.n_objects

        self.obs_shape = {'unary': (self.obj_n, self.n_attr),
                          'binary': (self.obj_n, self.obj_n, len(self.rel_deter_func))}
        self.spatial_tensors = None
        self.prev = None

    def extract_dense_attributes(self, data):
        # Compute the sum of attributes along the last dimension
        attribute_sum = np.sum(data, axis=2)

        non_zero_count = np.count_nonzero(attribute_sum)
        if non_zero_count != self.obj_n:
            print( f'object mismatch: {non_zero_count} non_zerocount but {self.obj_n} objects')
        # Find the indices of non-zero attribute sums
        non_zero_indices = np.nonzero(attribute_sum)

        # Filter out elements whose attributes sum to zero
        filtered_data = data[non_zero_indices]

        attribute_vectors = []

        for i in range(self.n_attr):
            attribute_vectors.append(np.reshape(filtered_data[:,i], non_zero_count))
        assert len(attribute_vectors[0]) == self.obj_n, f'{len(attribute_vectors[0])} attribute vectors but {self.obj_n} objects'
        return attribute_vectors

    def extract_attributes(self, data):
            # Get the size of the grid and number of attributes
            n_rows, n_cols, n_attr = data.shape

            # Initialize a list of attribute vectors, each with length n_rows * n_cols
            attribute_vectors = []

            for i in range(n_attr):
                attribute_vectors.append(np.reshape(data[:, :, i], (n_rows * n_cols)))
            return attribute_vectors

    def img2vkb(self, img, direction=None):
        """
        Takes in an RGB img (n+2, n+2, 3) and returns vectorized attributes
        Parameters
        ----------
        img : image of the environment
        direction :

        Returns
        unary_t (attribute per entity), binary_t (relation between entities)
        -------

        """
        img = img.astype(np.int32)
        # data = filter_non_zero_elements(img) if self.dense else img

        if self.dense:
            unary_tensors = self.extract_dense_attributes(img)
        else:
            unary_tensors = self.extract_attributes(img)
        objs = []
        for y, row in enumerate(img):
            for x, pixel in enumerate(row):
                # print(pixel)
                if self.dense and np.sum(pixel)==0:
                    continue
                obj = GridObject(x,y)
                objs.append(obj)

        if not self.spatial_tensors or self.dense:
            # create spatial tensors that gives for every rel. det rule a binary indicator between the entities
            self.spatial_tensors = [np.zeros([len(objs), len(objs)]) for _ in range(len(self.rel_deter_func))] # 14  81x81 vectors for each relation
            # self.relational_vectors = [np.zeros(len(self.rel_deter_func)) for _ in range(self.n_edges)]
            for obj_idx1, obj1 in enumerate(objs):
                for obj_idx2, obj2 in enumerate(objs):
                    direction_vec = DIR_TO_VEC[1]
                    for rel_idx, func in enumerate(self.rel_deter_func):
                        if func(obj1, obj2, direction_vec):
                            self.spatial_tensors[rel_idx][obj_idx1, obj_idx2] = 1.0
                            # self.relational_vectors[][rel_idx]

        self.prev = self.spatial_tensors

        binary_tensors = torch.tensor(self.spatial_tensors)
        if np.array_equal(self.prev, binary_tensors):
            pass
        unary_t = np.stack(unary_tensors, axis=-1)
        if self.rgcn:
            gd = to_gd(binary_tensors, nb_objects=self.obj_n)
        else:
            x = torch.arange(self.obj_n).view(-1, 1)
            edge_list = generate_directed_edge_list(self.obj_n)
            edge_attr = create_edge_attributes(objs=objs, edge_list=edge_list, rel_rules= self.rel_deter_func)
            gd = GeometricData(x=x, edge_index=edge_list, edge_attr=edge_attr)
        return unary_t, gd

    def observation(self, obs):
        """
        Wrapper that customizes the default observation space.
        Is called also when environment is reset!
        Parameters
        ----------
        obs : Observation from the default environment

        Returns
        -------

        """
        #obs = obs.copy()
        for ob in obs:
            spatial_VKB = self.img2vkb(ob['image'])
            ob['unary_tensor'], ob['binary_tensor'] = spatial_VKB
        return obs

    def id_to_rule_list(self,background_id):
        if background_id in ["b0", "nolocal"]:
            rel_deter_func = [is_left, is_right, is_front, is_back, is_aligned, is_close]
        elif background_id in ["t0", "any"]:
            rel_deter_func = [is_any]
        elif background_id == "b1":
            rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj]
        elif background_id in ["b2", "local"]:
            rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj]
        elif background_id in ["b3", "all"]:
            rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, is_aligned, is_close]
        elif background_id == "b4":
            rel_deter_func = [is_left, is_right, is_front, is_back]
        elif background_id == "b5":
            rel_deter_func = [is_top_adj, is_left_adj, is_down_adj, is_right_adj]
        elif background_id in ["b6", "noremote"]:
            rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_aligned, is_close]
        elif background_id in ["b7", "noauxi"]:
            rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back]
        elif background_id in ["b8"]:
            rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, fan_top, fan_down,
                                   fan_left, fan_right]
        elif background_id in ["b9"]:
            rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, top_right, top_left,
                                   down_left, down_right]
        else:
            rel_deter_func = None
        return rel_deter_func

def to_gd(spatial_tensors: torch.Tensor, nb_objects) -> GeometricData:
    """
    takes batch of adjacency geometric data and transforms it to a GeometricData object for torch.geometric

    Parameters
    --------
    data : heterogeneous adjacency matrix (nb_relations, nb_objects, nb_objects)
    """
    x = torch.arange(nb_objects).view(-1, 1)
    nz = torch.nonzero(spatial_tensors)

    # list of all edges and what relationtype they have
    edge_attr = nz[:, 0] # T(num_edges, )

    # list of node to node indicating an edge
    edge_index = nz[:, 1:].T # T(2, num_edges)
    return GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)




def rotate_vec2d(vec, degrees):
    """
    rotate a vector anti-clockwise
    :param vec:
    :param degrees:
    :return:
    """
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R@vec

def offset2idx_offset(x, y, width):
    return y*width+x

def is_self(obj1, obj2, _ )->bool:
    if obj1 == obj2:
        return True
    else:
        return False

def is_any(obj1, obj2, _ )->bool:
    if obj1 != obj2:
        return True
    else:
        return False
def is_front(obj1, obj2, direction_vec)->bool:
    diff = obj2.pos - obj1.pos
    return diff@direction_vec > 0.1


def is_back(obj1, obj2, direction_vec)->bool:
    diff = obj2.pos - obj1.pos
    return diff@direction_vec < -0.1


def is_left(obj1, obj2, direction_vec)->bool:
    left_vec = rotate_vec2d(direction_vec, -90)
    diff = obj2.pos - obj1.pos
    return diff@left_vec > 0.1

def is_right(obj1, obj2, direction_vec)->bool:
    left_vec = rotate_vec2d(direction_vec, 90)
    diff = obj2.pos - obj1.pos
    return diff@left_vec > 0.1

def is_close(obj1, obj2, direction_vec=None)->bool:
    distance = np.abs(obj1.pos - obj2.pos)
    return np.sum(distance)==1

def is_aligned(obj1, obj2, direction_vec=None)->bool:
    diff = obj2.pos - obj1.pos
    return np.any(diff==0)

def is_top_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.x==obj2.x and obj1.y==obj2.y+1

def is_left_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.y==obj2.y and obj1.x==obj2.x-1

def is_top_left_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.y==obj2.y+1 and obj1.x==obj2.x-1

def is_top_right_adj(obj1, obj2, direction_vec=None)->bool:
    return obj1.y==obj2.y+1 and obj1.x==obj2.x+1

def is_down_adj(obj1, obj2, direction_vec=None)->bool:
    return is_top_adj(obj2, obj1)

def is_right_adj(obj1, obj2, direction_vec=None)->bool:
    return is_left_adj(obj2, obj1)

def is_down_right_adj(obj1, obj2, direction_vec=None)->bool:
    return is_top_left_adj(obj2, obj1)

def is_down_left_adj(obj1, obj2, direction_vec=None)->bool:
    return is_top_right_adj(obj2, obj1)

def top_left(obj1, obj2, direction_vec)->bool:
    return (obj1.x-obj2.x) <= (obj1.y-obj2.y)

def top_right(obj1, obj2, direction_vec)->bool:
    return -(obj1.x-obj2.x) <= (obj1.y-obj2.y)

def down_left(obj1, obj2, direction_vec)->bool:
    return top_right(obj2, obj1, direction_vec)

def down_right(obj1, obj2, direction_vec)->bool:
    return top_left(obj2, obj1, direction_vec)

def fan_top(obj1, obj2, direction_vec)->bool:
    top_left = (obj1.x-obj2.x) <= (obj1.y-obj2.y)
    top_right = -(obj1.x-obj2.x) <= (obj1.y-obj2.y)
    return top_left and top_right

def fan_down(obj1, obj2, direction_vec)->bool:
    return fan_top(obj2, obj1, direction_vec)

def fan_right(obj1, obj2, direction_vec)->bool:
    down_left = (obj1.x-obj2.x) >= (obj1.y-obj2.y)
    top_right = -(obj1.x-obj2.x) <= (obj1.y-obj2.y)
    return down_left and top_right

def fan_left(obj1, obj2, direction_vec)->bool:
    return fan_right(obj2, obj1, direction_vec)


if __name__ == "__main__":
    from MABoxWorld.environments.box import BoxWorldEnv
    env = BoxWorldEnv(
        players=2,
        field_size=(5,5),
        num_colours=5,
        goal_length=2,
        sight=5,
        max_episode_steps=500,
        grid_observation=True,
        simple=False,
        deterministic= True,
    )
    env = AbsoluteVKBWrapper(env, dense=True)
    obs = env.reset()
    render= True
    done = False
    if render:
        env.render()
        # pygame.display.update()  # update window
        time.sleep(0.5)

    while not done:
        # for i in range(100):
        actions = env.action_space.sample()
        nobs, nreward, ndone, _ = env.step(actions)

        print('printing obs', nobs)
        # print('player pos', env.players[0].position, '----', env.players[1].position )
        # nobs, nreward, ndone, _ = env.step((1,1))
        if sum(nreward) != 0:
            print(nreward)

        if render:
            env.render()
            # pygame.display.update()  # update window
            time.sleep(0.5)

        done = np.all(ndone)
        pygame.event.pump()  # process event queue
    # print(env.players[0].score, env.players[1].score)


def generate_directed_edge_list(n):
    edge_list = []
    for src in range(0, n):
        for tgt in range(0, n):
            if src != tgt:
                edge_list.append((src, tgt))
    return edge_list

def create_edge_attributes(objs, edge_list, rel_rules):
    edge_attributes = np.zeros((len(edge_list), len(rel_rules)))
    direction_vec = DIR_TO_VEC[1]
    for idx, edge in enumerate(edge_list):
        src_idx, tgt_idx = edge
        src, tgt = objs[src_idx], objs[tgt_idx]
        attribute = [1 if rule(src, tgt, direction_vec) else 0 for rule in rel_rules]
        edge_attributes[idx,:] = attribute
    edge_attributes = torch.tensor(edge_attributes)
    return edge_attributes