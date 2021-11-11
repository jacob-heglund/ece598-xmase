from gym_multigrid.multigrid import *
from gym_multigrid.watcher import *
from gym import error, spaces, utils

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

        self.world = World

        agents = []
        if not watcher:
            agents.append(Collector(self.world, 2, view_size=7))
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

        for number, index, reward in zip(self.num_goals, self.goals_index, self.goals_reward):
            for i in range(number):
                self.place_obj(Goal(self.world, index))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)


    # def _reward(self, i, rewards, reward=1):
    #     """
    #     Compute the reward to be given upon success
    #     """
    #     for j,a in enumerate(self.agents):
    #         if a.index==i or a.index==0:
    #             rewards[j]+=reward
    #         if self.zero_sum:
    #             if a.index!=i or a.index==0:
    #                 rewards[j] -= reward

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info


class CollectGameSingleAgent(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_goals=[1],
        agents_index = [2],
        goals_index=[0])

class CollectGameWatcher(CollectGameEnv):
    def __init__(self):
        super().__init__(size=10,
        num_goals=[1],
        agents_index = [1,2],
        goals_index=[0],
        watcher=True)


