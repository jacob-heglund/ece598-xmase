from gym_multigrid.multigrid import *
from gym_multigrid.watcher import *
from gym import error, spaces, utils

class CollectGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [],
        balls_index=[],
        balls_reward=[],
        zero_sum = False,
        view_size=7,
        watcher=False
    ):
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.num_collected = 0

        self.world = World

        if not watcher:
            agents = []
            for i in agents_index:
                agents.append(Agent(self.world, i, view_size=view_size))
        else:
            agents = []
            #for i in agents_index:
            agents.append(Watcher(self.world, 1))
            agents.append(Collector(self.world, 2))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
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

        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                self.place_obj(Ball(self.world, index, reward))

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
        if self.num_collected == self.num_balls:
            done = True
        return obs, rewards, done, info


class CollectGame4HEnv10x10N2(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[5],
        agents_index = [1,2,3],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=True)

class CollectGameWatcher(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_balls=[1],
        agents_index = [1,2],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=True,
        watcher=True)
        self.action_set_watcher=SmallActions
        self.action_space_watcher = spaces.Discrete(len(self.action_set_watcher.available))

