from gym_multigrid.multigrid import *

class CTFEnv(MiniGridEnv):
    """
    CTF environment
    """
    def __init__(
            self,
            width,
            height,
            flag_pos,
            flag_index,
            agent_index,
            floor_index,
            view_size=3
                 ):

        self.flag_pos = flag_pos
        self.flag_index= flag_index
        self.floor_index = floor_index

        self.world = World

        agents = []
        for i in agent_index:
            agents.append(Agent(self.world, i, view_size=view_size))


        super().__init__(
            width=width,
            height=height,
            max_steps = 150,
            agents = agents,
            agent_view_size = view_size
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for i in range(len(self.flag_pos)):
            self.place_obj(Flag(self.world,self.flag_index[i]), top=self.flag_pos[i], size=[1,1])

        for i in range(0,19):
            for j in range(1,9):
                self.put_obj(Colored_Floor(self.world, 1), i, j)
            for j in range(10,18):
                self.put_obj(Colored_Floor(self.world, 2), i, j)

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info

class CTFEnv2v2(CTFEnv):
    def __init__(self):
        super().__init__(
            height = 20,
            width = 20,
            flag_pos = [[1,10], [18,10]],
            flag_index = [1,2],
            agent_index = [1,1,2,2],
            floor_index = [1,2]
        )

