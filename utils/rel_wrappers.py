from Aurora.agent.util import rotate_vec2d
from Aurora.environment.box.boxworld_gen import all_colors
from gym_minigrid.minigrid import * # imports DIR_TO_VEC as [array([1, 0]), array([0, 1]), array([-1,  0]), array([ 0, -1])]

#OBJECTS = ind_dict2list(OBJECT_TO_IDX)

class GridObject():
    "object is specified by its location"
    def __init__(self, x, y, object_type=[]):
        self.x = x
        self.y = y
        self.type = object_type
        self.attributes = object_type

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def name(self):
        return str(self.type)+"_"+str(self.x)+str(self.y)


def parse_object(x:int, y:int, feature)->GridObject:
    """
    :param x: column position
    :param y: row position
    :param feature: feature should just be a single digit here, i.e. the color of the box
    :return:
    """
    # TODO this needs to change as we currently already have the color index
    #obj_type = [str(color2index[tuple(feature)])] # takes in RBG color from pixel and returns color index
    obj_type = [str(feature)] #now it should just be the index of the color that the object has
    obj = GridObject(x, y, object_type=obj_type)
    return obj


from random import randint
class AbsoluteVKBWrapper(gym.core.ObservationWrapper):
    """
    Add a vkb key-value pair, which represents the state as a vectorised knowledge base.
    Entities are objects in the gird-world, predicates represents the properties of them and relations between them.
    """
    def __init__(self, env, background_id="b3"):
        super().__init__(env)

        background_id = background_id[:2]
        self.attributes = [str(i) for i in range(len(all_colors))]
        self.env_type = "boxworld"
        self.nullary_predicates = []
        self.unary_predicates = self.attributes
        self.background_id = background_id
        if background_id in ["b0", "nolocal"]:
            self.rel_deter_func = [is_left, is_right, is_front, is_back, is_aligned, is_close]
        elif background_id == "b1":
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj]
        elif background_id in ["b2", "local"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj]
        elif background_id in ["b3", "all"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, is_aligned, is_close]
        elif background_id == "b4":
            self.rel_deter_func = [is_left, is_right, is_front, is_back]
        elif background_id == "b5":
            self.rel_deter_func = [is_top_adj, is_left_adj, is_down_adj, is_right_adj]
        elif background_id in ["b6", "noremote"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_aligned, is_close]
        elif background_id in ["b7", "noauxi"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back]
        elif background_id in ["b8"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, fan_top, fan_down,
                                   fan_left, fan_right]
        elif background_id in ["b9"]:
            self.rel_deter_func = [is_top_adj, is_left_adj, is_top_right_adj, is_top_left_adj,
                                   is_down_adj, is_right_adj, is_down_left_adj, is_down_right_adj,
                                   is_left, is_right, is_front, is_back, top_right, top_left,
                                   down_left, down_right]

        # number of objects/entities are the number of cells on the grid
        self.obj_n = np.prod(env.observation_space["image"].shape[:-1]) #physical entities # TODO needs to be adjusted: we do not have obs['image'] yet
        self.nb_all_entities = self.obj_n
        self.obs_shape = [(len(self.nullary_predicates)), (self.obj_n, len(self.attributes)),
                       (self.obj_n, self.obj_n, len(self.rel_deter_func))] # TODO remove nullary_predicates?
        self.spatial_tensors = None

    def img2vkb(self, img, direction=None):
        """
        Takes in an RGB img (n+2, n+2, 3) and returns vectorized attributes
        Parameters
        ----------
        img : image of the environment
        direction :

        Returns
        nullary_t (direction) , unary_t (attribute per entity), binary_t (relation between entities)
        -------

        """
        img = img.astype(np.int32)
        unary_tensors = [np.zeros(self.obj_n) for _ in range(len(self.unary_predicates))] # so this is a list of binary vectors for each color, indicating what entities have the colour
        objs = []
        for y, row in enumerate(img):
            for x, pixel in enumerate(row):
                obj = parse_object(x, y, pixel)
                objs.append(obj)

        nullary_tensors = []

        for obj_idx, obj in enumerate(objs):
            for p_idx, p in enumerate(self.attributes):
                if p in obj.attributes:
                    # adds feature, e.g. colour as unary tensor
                    unary_tensors[p_idx][obj_idx] = 1.0

        if not self.spatial_tensors:
            # create spatial tensors that gives for every rel. det rule a binary indicator between the entities
            self.spatial_tensors = [np.zeros([len(objs), len(objs)]) for _ in range(len(self.rel_deter_func))] # 14  81x2 vectors for each relation
            for obj_idx1, obj1 in enumerate(objs):
                for obj_idx2, obj2 in enumerate(objs):
                    direction_vec = DIR_TO_VEC[1] # TODO I'd say it's down - CHANGED IT TO 1 INSTEAD OF 3
                    for rel_idx, func in enumerate(self.rel_deter_func):
                        if func(obj1, obj2, direction_vec):
                            self.spatial_tensors[rel_idx][obj_idx1, obj_idx2] = 1.0

        binary_tensors = self.spatial_tensors # these vectors should change every time with the environment
        nullary_t, unary_t, binary_t = nullary_tensors, np.stack(unary_tensors, axis=-1), np.stack(binary_tensors, axis=-1)
        return nullary_t, unary_t, binary_t

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
        obs = obs.copy()
        spatial_VKB = self.img2vkb(obs["image"])
        obs["VKB"] = spatial_VKB # additional information in form of features and no external VKB for us
        return obs




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
    from MABoxWorld.environments.rel_box import BoxWorldEnv
    env = BoxWorldEnv(
        players=1,
        field_size=(5,5),
        num_colours=5,
        goal_length=2,
        sight=5,
        max_episode_steps=500,
        grid_observation=True,
        simple=True,
        relational = False,
        deterministic= False,
    )
    obs = env.reset()
    print(obs[0].shape)
    #env = AbsoluteVKBWrapper(env)

