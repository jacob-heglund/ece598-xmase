from gym_multigrid.multigrid import *
from gym_multigrid.watcher import *
from gym import error, spaces, utils
import random

class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to reach the goal
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_goals=[],
        agents_index = [],
        goals_index=[],
        goals_reward=[],
        zero_sum = False,
        view_size=7,
        watcher=False
    ):
        self.num_goals = num_goals
        self.goals_index = goals_index
        self.goals_reward = goals_reward
        self.zero_sum = zero_sum
        self.num_collected = 0

        self.world = World

        agents = []
        if not watcher:
            for i in agents_index:
                agents.append(Agent(self.world, i, view_size=view_size))
        else:
            agents.append(Watcher(self.world, 1))
            agents.append(Collector(self.world, 2))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 1000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size
        )


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for number, index in zip(self.num_goals, self.goals_index):
            for i in range(number):
                pos = self.place_obj(Goal(self.world, index, color=4))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)
                    self.num_collected += 1

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        if self.num_collected == self.num_goals:
            done = True
        return obs, rewards, done, info


class CollectGameSingleAgent(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_goals=[1],
        agents_index = [2],
        goals_index=[0])

class CollectGameRat(CollectGameEnv):
    def __init__(self):
        super().__init__(size=5,
        num_goals=[1],
        agents_index = [1],
        goals_index=[0],
        view_size=3)
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls and make a T
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # player starts at bottom of T
        for a in self.agents:
            pos = np.array((width - 1, int(width / 2)))
            self.grid.set(*pos, a)
            a.init_pos = pos
            a.pos = pos
            a.init_dir = 3
            a.dir = 3

        # goal is at top left or top right
        for number, index in zip(self.num_goals, self.goals_index):
            for i in range(number):
                k = random.randint(0, 1)
                # left
                if k == 0:
                    col = int(width / 2) - 1
                # right
                else:
                    col = int(width / 2) + 1
                pos = np.array((0, col))
                obj = Goal(self.world, index)
                self.grid.set(*pos, obj)
                if obj is not None:
                    obj.init_pos = pos
                    obj.cur_pos = pos

        # indicator is somewhere in front of the agent in the stem of T
        col = self.np_random.randint(0, width - 2)
        pos = np.array((0, col))
        
        # 2 is blue, 4 is yellow
        k = random.randint(0, 1)
        idx = k * 2 + 2
        
        light = Light(self.world, color=self.world.IDX_TO_COLOR[idx])
        self.grid.set(*pos, light)
        light.init_pos = pos
        light.cur_pos = pos

class CollectGameWatcher(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_goals=[1],
        agents_index = [1,2],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=True,
        watcher=True)
        self.action_set_watcher=SmallActions
        self.action_space_watcher = spaces.Discrete(len(self.action_set_watcher.available))