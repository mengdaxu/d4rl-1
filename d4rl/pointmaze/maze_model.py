""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random
import mujoco_py
from d4rl.pointmaze.maze_layouts import U_MAZE

WALL = 10
EMPTY = 11
GOAL = 12
CHASER = 13
BOX = 14


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == 'C':
                maze_arr[w][h] = CHASER
            elif tile == 'B':
                maze_arr[w][h] = BOX
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(
        type="2d",
        name="groundplane",
        builtin="checker",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        # rgb1=np.random.rand(3),
        # rgb2=np.random.rand(3),
        width=100,
        height=100)
    asset.texture(name="skybox",
                  type="skybox",
                  builtin="gradient",
                  rgb1=".4 .6 .8",
                  rgb2="0 0 0",
                  width="800",
                  height="800",
                  mark="random",
                  markrgb="1 1 1")
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="box", rgba=".6 .3 .3  1")
    # asset.material(name="box", rgba="0.5 0.3 0.2 1")
    asset.material(name="target", rgba="0 1 0 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',
                   size="40 40 0.25",
                   pos="0 0 -0.1",
                   type="plane",
                   contype=1,
                   conaffinity=0,
                   material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2, 1.2, 0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 0.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0, 0.0, 0], size=0.2, rgba='1.0 1.0 1.0 1')
    particle.joint(name='ball_x', type='slide', pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name='ball_y', type='slide', pos=[0, 0, 0], axis=[0, 1, 0])

    if "C" in maze_str:
        # print("add chaser")
        chaser = worldbody.body(name='chaser', pos=[1.2, 1.2, 0])
        chaser.geom(name='chaser_geom', type='sphere', size=0.1, rgba='.6 .3 .3  1', contype=1)
        chaser.site(name='chaser_site', pos=[0.0, 0.0, 0], size=0.2, rgba='.6 .3 .3  1')
        # chaser.site(name='chaser_site', pos=[0.0, 0.0, 0], size=0.2, rgba='0.5 0.3 0.2 1')
        chaser.joint(name='chaser_x', type='slide', pos=[0, 0, 0], axis=[1, 0, 0])
        chaser.joint(name='chaser_y', type='slide', pos=[0, 0, 0], axis=[0, 1, 0])

    if "G" in maze_str:
        worldbody.site(name='target_site', pos=[0.0, 0.0, 0], size=0.2, material='target')

    width, height = maze_arr.shape

    box_count = 0
    for w in range(width):
        for h in range(height):
            if maze_arr[w, h] == BOX:
                worldbody.site(name='box_%d_site' % (box_count), pos=[10, 10, 0], size=0.2, material='box')
                box_count += 1

    for w in range(width):
        for h in range(height):
            if maze_arr[w, h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d' % (w, h),
                               material='wall',
                               pos=[w + 1.0, h + 1.0, 0],
                               size=[0.5, 0.5, 0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    # for topdown rendering
    # worldbody.camera(mode="fixed", name="birdview", pos="11.5 11.5 29.0", quat="0.7071 0 0 0.7071")
    # worldbody.camera(mode="fixed", name="birdview", pos="21.5 21.5 49.0", quat="0.7071 0 0 0.7071")
    if "C" in maze_str:
        actuator.motor(joint="chaser_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
        actuator.motor(joint="chaser_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    AGENT_CENTRIC_RES = 32

    def __init__(self,
                 maze_spec=U_MAZE,
                 reward_type='dense',
                 reset_target=False,
                 reset_box=False,
                 agent_centric_view=False,
                 dummy_goal=False,
                 be_close=False,
                 reward_scale=1,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.reset_target = reset_target
        self.reset_box = reset_box
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.be_close = be_close
        self.agent_centric_view = agent_centric_view
        # self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations = list(zip(*np.where(self.maze_arr != WALL)))
        self.reset_locations.sort()

        self.chaser_exist = False
        self.goal_exist = False
        self.box_exist = False
        self.dummy_goal = dummy_goal
        self.reward_scale = reward_scale
        if 'C' in self.str_maze_spec:
            self.chaser_exist = True
        if 'G' in self.str_maze_spec:
            self.goal_exist = True

        self._target = np.array([0.0, 0.0])
        self._box = None
        # self._random_state = np.random.RandomState(0)

        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        if self.dummy_goal or self.goal_exist:
            self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
            if len(self.goal_locations) == 1:
                self.set_target(self.goal_locations[0])
            elif len(self.goal_locations) > 1:
                raise ValueError("More than 1 goal specified!")
            else:
                # If no goal, use the first empty tile
                self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))

            self.empty_and_goal_locations = self.reset_locations + self.goal_locations
        else:
            self.empty_and_goal_locations = self.reset_locations

        if 'B' in self.str_maze_spec:
            self.box_exist = True
            self.box_count = self.str_maze_spec.count('B')
            self.set_box()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        info = {'hit': False, 'reach': False}
        if self.reward_type == 'sparse':
            reward = 0
            if self.goal_exist:
                distance = np.linalg.norm(ob[0:2] - self._target)
                if distance <= 0.6:
                    reward = self.reward_scale
                    info['reach'] = True
            if self.box_exist:
                distance = np.linalg.norm(ob[0:2] - self._box, axis=-1)
                if (distance <= 0.6).any():
                    info['reach'] = False
                    info['hit'] = True
        elif self.reward_type == 'dense':
            if self.goal_exist:
                reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
            else:
                raise NotImplementedError
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        done = False
        return ob, reward, done, info

    def render(self, mode, *args, **kwargs):
        if self.agent_centric_view:
            if 'width' not in kwargs and 'height' not in kwargs:
                kwargs['width'] = self.AGENT_CENTRIC_RES
                kwargs['height'] = self.AGENT_CENTRIC_RES
        return super().render(mode, *args, **kwargs)

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_target(self):
        return self._target

    def gridify_state(self, state):
        return (int(round(state[0])), int(round(state[1])))

    @property
    def _agent_index(self):
        return slice(0, 2)

    @property
    def _chaser_index(self):
        return slice(2, 4)

    @property
    def _chaser(self):
        return self.sim.data.qpos[self._chaser_index]

    def set_target(self, target_location=None, agent_location=None):
        idx = None
        close = True
        if target_location is None:
            while close:
                idx = self.np_random.choice(len(self.empty_and_goal_locations))
                reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
                target_location = reset_location  #+ self.np_random.uniform(low=-.1, high=.1, size=2)
                if agent_location is None or np.linalg.norm(agent_location - target_location) > 5:
                    close = False
        self._target = target_location

        pos = self.gridify_state(self._target)
        self.maze_arr[pos] = GOAL

        return idx

    def set_box(self, box_location=None, avoid_index=[]):
        if box_location is None:
            box_location = np.zeros((self.box_count, 2))
            for i in range(self.box_count):
                occupy = True
                iter = 0
                while occupy:
                    iter += 1
                    idx = self.np_random.choice(len(self.empty_and_goal_locations))
                    reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
                    if i > 0:
                        dis = np.linalg.norm(reset_location - box_location[:i], axis=-1)
                    else:
                        dis = np.array([np.inf])
                    if idx not in avoid_index and (dis > 4).all():
                        occupy = False

                    if iter > 100:
                        # print("fail to find")
                        break
                box_location[i] = reset_location  #+ self.np_random.uniform(low=-.1, high=.1, size=2)
                pos = self.gridify_state(box_location[i])
                # self.maze_arr[pos] = BOX

        for i in range(self.box_count):
            pos = self.gridify_state(box_location[i])
            self.maze_arr[pos] = BOX

        self._box = box_location

    def set_marker(self):
        if self.goal_exist:
            self.data.site_xpos[self.model.site_name2id('target_site')] = np.array(
                [self._target[0] + 1, self._target[1] + 1, 0.0])

        if self.box_exist:
            for i in range(self.box_count):
                self.data.site_xpos[self.model.site_name2id('box_%d_site' % (i))] = np.array(
                    [self._box[i, 0] + 1, self._box[i, 1] + 1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        for i in self.empty_and_goal_locations:
            self.maze_arr[i] = EMPTY
        agent_idx, chaser_idx = self.np_random.choice(len(self.empty_and_goal_locations), 2)
        agent_reset_location = np.array(self.empty_and_goal_locations[agent_idx]).astype(self.observation_space.dtype)
        if self.be_close:
            distance = np.linalg.norm(agent_reset_location - self.empty_and_goal_locations, axis=1)
            chaser_idx = np.argsort(distance)[int(len(self.empty_and_goal_locations) * 1 / 10)]
        chaser_reset_location = np.array(self.empty_and_goal_locations[chaser_idx]).astype(self.observation_space.dtype)
        if self.chaser_exist:
            qpos = np.concatenate([agent_reset_location, chaser_reset_location])
            # + self.np_random.uniform(    low=-.1, high=.1, size=self.model.nq)
        else:
            qpos = agent_reset_location  # + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            target_index = self.set_target(agent_location=qpos[self._agent_index])
        if self.reset_box:
            self.set_box(avoid_index=[agent_idx, target_index])
        return self._get_obs()

    def reset_to_location(self, location, seed=None):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location  # + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel  # + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        if self.agent_centric_view:
            self.viewer.cam.type = mujoco_py.generated.const.CAMERA_TRACKING
            self.viewer.cam.distance = 5.0
        else:
            self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.trackbodyid = 1  # id of the body to track ()
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis