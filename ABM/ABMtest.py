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
        for i in range(1, self.move_times+1):
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
        if len(cellmates)>1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1


class BusinessModel(mesa.Model):
    """1. Business agents only move to and locate in grids of Tokyo instead of rectangle.
       2. Trip times of each agent is based on
       3. Trip patterns of each agent will be extracted from ETC data.
       4. Destination is restricted to grids with POIs."""

    def __init__(self, N, admin_grids_df, agent_freq, ori_type_prob_df, poi_df):
        self.num_agents = N
        self.admin_grids_df = admin_grids_df
        self.admin_grids = None
        self.agent_freq = agent_freq
        self.ori_type_prob_df = ori_type_prob_df
        self.poi_df = poi_df
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

    # def create_agents(self):
    #     # Add a new column for the cumulative probability of each grid for first moves
    #     self.admin_grids_df['cum_prob_first'] = self.admin_grids_df['firstgridprob'].cumsum()
    #     # Add a new column for the cumulative probability of each grid for subsequent moves
    #     self.admin_grids_df['cum_prob_des'] = self.admin_grids_df['desgridprob'].cumsum()
    #     # Read OriTypeProb data
    #     type_prob = self.ori_type_prob_df.set_index('Type')['Percentage'].to_dict()
    #
    #     for i in range(self.num_agents):
    #         # Decide how many business trips the agent will take
    #         trip_times = np.random.choice(self.agent_freq['times'], p=self.agent_freq['frequency'])
    #
    #         # Create type of vaild grids before agent's first move
    #         a = BusinessAgent(i, self, trip_times, self.admin_grids)
    #         a_type = np.random.choice(list(type_prob.keys()), p=list(type_prob.values()))
    #         print("Agent", a.unique_id, "Original POI type:", a_type)
    #         a.type = a_type
    #         a.destination_types = []  # Initialize empty list to store destination POI types
    #
    #         # Create location of vaild grids before agent's first move
    #         while True:
    #             rand_num = np.random.random()
    #             idx = self.admin_grids_df['cum_prob_first'].searchsorted(rand_num)
    #             admin_grid = self.admin_grids_df.iloc[idx]
    #             x, y = admin_grid[['x', 'y']].values
    #             # Small question that agents are placed into different grids at first
    #             if self.grid.is_cell_empty((x, y)):
    #                 self.grid.place_agent(a, (x, y))
    #                 break
    #
    #         # Move the agents
    #         for j in range(1, trip_times + 1):
    #             print("Agent", a.unique_id, "of", j, "move times")
    #             dest_types = []
    #             while True:
    #                 rand_num = np.random.random()
    #                 idx = self.admin_grids_df['cum_prob_des'].searchsorted(rand_num)
    #                 admin_grid = self.admin_grids_df.iloc[idx]
    #                 x, y = admin_grid[['x', 'y']].values
    #                 if (x, y) != a.pos:
    #                     poi_subset = self.poi_df[(self.poi_df['x'] == x) & (self.poi_df['y'] == y)]
    #                     if not poi_subset.empty:
    #                         dest_type = poi_subset.sample(n=1)['DesType'].iloc[0]
    #                     else:
    #                         dest_type = ''
    #                     print("Agent", a.unique_id, "destination POI", dest_type)
    #                     dest_types.append(dest_type)
    #                     self.grid.move_agent(a, (x, y))
    #                     break
    #
    #             # Update destination types
    #             a.destination_types = dest_types

    def create_agents(self):
        # Add a new column for the cumulative probability of each grid for first moves
        self.admin_grids_df['cum_prob_first'] = self.admin_grids_df['firstgridprob'].cumsum()
        # Add a new column for the cumulative probability of each grid for subsequent moves
        self.admin_grids_df['cum_prob_des'] = self.admin_grids_df['desgridprob'].cumsum()
        # Read OriTypeProb data
        type_prob = self.ori_type_prob_df.set_index('Type')['Percentage'].to_dict()

        # Use vectorized approach to update agent's destination_types attribute
        dest_type_subset = self.poi_df.set_index(['x', 'y'])[['DesType']]
        dest_type_dict = dest_type_subset.to_dict()['DesType']

        for i in range(self.num_agents):
            # Decide how many business trips the agent will take
            trip_times = np.random.choice(self.agent_freq['times'], p=self.agent_freq['frequency'])

            # Create type of vaild grids before agent's first move
            a = BusinessAgent(i, self, trip_times, self.admin_grids)
            a_type = np.random.choice(list(type_prob.keys()), p=list(type_prob.values()))
            print("Agent", a.unique_id, "Original POI type:", a_type)
            a.type = a_type

            # Use numpy.random.choice to select a random row directly
            self.admin_grids_df['firstgridprob'] = self.admin_grids_df['firstgridprob'] / self.admin_grids_df['firstgridprob'].sum()
            first_move_row = np.random.choice(self.admin_grids_df.index, p=self.admin_grids_df['firstgridprob'])
            x, y = self.admin_grids_df.loc[first_move_row, ['x', 'y']].values
            self.grid.place_agent(a, (x, y))

            dest_types = []
            # Use numpy.random.choice to select a random row directly
            for j in range(1, trip_times + 1):
                print("Agent", a.unique_id, "of", j, "move times")
                des_move_row = np.random.choice(self.admin_grids_df.index, p=self.admin_grids_df['desgridprob'])
                x, y = self.admin_grids_df.loc[des_move_row, ['x', 'y']].values
                if (x, y) != a.pos:
                    dest_type = dest_type_dict.get((x, y), '')
                    print("Agent", a.unique_id, "destination POI", dest_type)
                    dest_types.append(dest_type)
                    self.grid.move_agent(a, (x, y))

            # Update destination types
            a.destination_types = dest_types

    def step(self):
        self.schedule.step()
