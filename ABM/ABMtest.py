import mesa
import numpy as np


class BusinessAgent(mesa.Agent):
    """Define the act and condition of business agents."""

    def __init__(self, unique_id, model, move_times, admin_grids):
        super().__init__(unique_id, model)
        self.wealth = 100
        self.admin_grids = admin_grids
        self.move_times = move_times

    def move(self):
        for i in range(self.move_times):
            while True:
                # Sample a destination grid with probability greater than 0
                dest_grids = self.admin_grids[self.admin_grids['desgridprob'] > 0]
                admin_grid = dest_grids.sample()
                x = admin_grid['x'].item()
                y = admin_grid['y'].item()
                if (x, y) != self.pos:
                    self.model.grid.move_agent(self, (x, y))
                    break

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other_agent = self.random.choice(cellmates)
            other_agent.wealth += 1
            self.wealth -= 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()


class BusinessModel(mesa.Model):
    """1. Business agents only move to and locate in grids of Tokyo instead of rectangle.
       2. Trip times of each agent is based on
       3. Trip patterns of each agent will be extracted from ETC data.
       4. Destination is restricted to grids with POIs."""

    def __init__(self, N, admin_grids_df, agent_freq, ori_type_prob_df):
        self.num_agents = N
        self.admin_grids_df = admin_grids_df
        self.admin_grids = None
        self.agent_freq = agent_freq
        self.ori_type_prob_df = ori_type_prob_df
        self.grid = mesa.space.MultiGrid(admin_grids_df['x'].max() + 1, admin_grids_df['y'].max() + 1, True)
        self.schedule = mesa.time.RandomActivation(self)
        # Load the valid_grids data
        self.load_admin_grids()
        # Create agents
        self.create_agents()

    def load_admin_grids(self):
        # Load the valid_grids data if not already loaded
        if self.admin_grids is None:
            self.admin_grids = self.admin_grids_df.copy()

    def create_agents(self):
        # Add a new column for the cumulative probability of each grid for first moves
        self.admin_grids_df['cum_prob_first'] = self.admin_grids_df['firstgridprob'].cumsum()
        # Add a new column for the cumulative probability of each grid for subsequent moves
        self.admin_grids_df['cum_prob_des'] = self.admin_grids_df['desgridprob'].cumsum()
        # Read OriTypeProb data
        type_prob = self.ori_type_prob_df.set_index('Type')['Percentage'].to_dict()

        # Decide how many business trips of each agent
        move_times = []
        for i in range(self.num_agents):
            trip_times = np.random.choice(self.agent_freq['times'], p=self.agent_freq['frequency'])
            move_times.append(trip_times)

        # Decide detailed trip patterns of the original business trips of each agent.
        for i in range(self.num_agents):
            a = BusinessAgent(i, self, move_times[i], self.admin_grids)
            print("")
            print("Agent", a.unique_id, "total move times:", a.move_times)
            self.schedule.add(a)
            # Add the agent to a valid grid cell for first move
            while True:
                rand_num = np.random.random()
                idx = self.admin_grids_df['cum_prob_first'].searchsorted(rand_num)
                admin_grid = self.admin_grids_df.iloc[idx]
                x = admin_grid['x'].item()
                y = admin_grid['y'].item()
                if self.grid.is_cell_empty((x, y)):
                    a_type = np.random.choice(list(type_prob.keys()), p=list(type_prob.values()))
                    print("Agent", a.unique_id, "Original POI type:", a_type)
                    a.type = a_type
                    self.grid.place_agent(a, (x, y))
                    break

            # Move the agent for subsequent moves
            for j in range(1, a.move_times+1):
                while True:
                    print("Agent", a.unique_id, "of", j, "move times")
                    rand_num = np.random.random()
                    idx = self.admin_grids_df['cum_prob_des'].searchsorted(rand_num)
                    admin_grid = self.admin_grids_df.iloc[idx]
                    x = admin_grid['x'].item()
                    y = admin_grid['y'].item()
                    if (x, y) != a.pos:
                        self.grid.move_agent(a, (x, y))
                        break
                    break

    def step(self):
        self.schedule.step()

