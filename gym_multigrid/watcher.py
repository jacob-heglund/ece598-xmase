from gym_multigrid.multigrid import *

class Watcher(Agent):
    def __init__(self, world, index=0, view_size=7):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        # self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.watcher = True
        self.comm = []

class Collector(Agent):
    def __init__(self, world, index=0, view_size=3):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.watcher = False
        self.comm = []

class Dummy(Agent):
    def __init__(self, world, index=0, view_size=3):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.watcher = False
        self.comm = []

class Rat(Agent):
    def __init__(self, world, index=0, view_size=1):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.watcher = False