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
    #
    #         # Use numpy.random.choice to select a random row directly
    #         self.admin_grids_df['firstgridprob'] = self.admin_grids_df['firstgridprob'] / self.admin_grids_df['firstgridprob'].sum()
    #         first_move_row = np.random.choice(self.admin_grids_df.index, p=self.admin_grids_df['firstgridprob'])
    #         x, y = self.admin_grids_df.loc[first_move_row, ['x', 'y']].values
    #         self.grid.place_agent(a, (x, y))
    #
    #         dest_types = []
    #         # Use numpy.random.choice to select a random row directly
    #         for j in range(1, trip_times + 1):
    #             # add
    #             des_move_row = np.random.choice(self.admin_grids_df.index, p=self.admin_grids_df['desgridprob'])
    #             x, y = self.admin_grids_df.loc[des_move_row, ['x', 'y']].values
    #             poi_subset = self.poi_df[
    #                 (self.poi_df['OriType'] == a_type) & (self.poi_df['x'] == x) & (self.poi_df['y'] == y)]
    #             if (x, y) != a.pos:
    #                 dest_volumes_subset = dict(zip(poi_subset['DesType'], poi_subset['volume']))
    #                 if len(dest_volumes_subset) > 0:
    #                     dest_volumes_list = [v for v in dest_volumes_subset.values()]
    #                     dest_volumes_norm = np.array(dest_volumes_list) / sum(dest_volumes_list)
    #                     dest_type_idx = np.random.choice(len(dest_volumes_list), p=dest_volumes_norm)
    #                     dest_type = list(dest_volumes_subset.keys())[dest_type_idx]
    #                 else:
    #                     dest_type = ''
    #                 print("Agent", a.unique_id, "of", j, "move times", "with destination POI of", dest_type)
    #                 dest_types.append(dest_type)
    #                 self.grid.move_agent(a, (x, y))
    #                 a_type = dest_types[-1]

    def create_agents(self):
        # Add a new column for the cumulative probability of each grid for first moves
        self.admin_grids_df['cum_prob_first'] = self.admin_grids_df['firstgridprob'].cumsum()
        # Add a new column for the cumulative probability of each grid for subsequent moves
        self.admin_grids_df['cum_prob_des'] = self.admin_grids_df['desgridprob'].cumsum()
        # Read OriTypeProb data
        type_prob = self.ori_type_prob_df.set_index('Type')['Percentage'].to_dict()

        # Generate all random selections for agents and trips
        trip_times_arr = np.random.choice(self.agent_freq['times'], size=self.num_agents,
                                          p=self.agent_freq['frequency'])
        a_types_arr = np.random.choice(list(type_prob.keys()), size=self.num_agents, p=list(type_prob.values()))
        first_move_rows = np.random.choice(self.admin_grids_df.index, size=self.num_agents,
                                           p=self.admin_grids_df['firstgridprob'] / self.admin_grids_df[
                                               'firstgridprob'].sum())
        des_move_rows = np.random.choice(self.admin_grids_df.index, size=(self.num_agents, trip_times_arr.max()),
                                         p=self.admin_grids_df['desgridprob'])

        # Loop through agents and apply selections
        for i in range(self.num_agents):
            trip_times = trip_times_arr[i]
            a_type = a_types_arr[i]

            # Create type of vaild grids before agent's first move
            a = BusinessAgent(i, self, trip_times, self.admin_grids)
            print("Agent", a.unique_id, "Original POI type:", a_type)
            a.type = a_type

            first_move_row = first_move_rows[i]
            # 问题：代理初始位置的坐标目前仅根据车辆数随机分布在所有东京格网中，需要进一步设置约束条件
            x, y = self.admin_grids_df.loc[first_move_row, ['x', 'y']].values
            self.grid.place_agent(a, (x, y))

            dest_types = []
            for j in range(1, trip_times + 1):
                des_move_row = des_move_rows[i, j - 1]
                x, y = self.admin_grids_df.loc[des_move_row, ['x', 'y']].values
                poi_subset = self.poi_df[
                    (self.poi_df['OriType'] == a_type) & (self.poi_df['x'] == x) & (self.poi_df['y'] == y)]
                if (x, y) != a.pos:
                    dest_volumes_subset = dict(zip(poi_subset['DesType'], poi_subset['volume']))
                    if len(dest_volumes_subset) > 0:
                        dest_volumes_list = [v for v in dest_volumes_subset.values()]
                        # 问题：起点POI类型和终点格网内包含的POI类型之间可能不存在volume，使posilibity分母为0，需要修改
                        # if sum(dest_volumes_list) == 0:
                        dest_volumes_norm = np.array(dest_volumes_list) / sum(dest_volumes_list)
                        dest_type_idx = np.random.choice(len(dest_volumes_list), p=dest_volumes_norm)
                        dest_type = list(dest_volumes_subset.keys())[dest_type_idx]
                    else:
                        dest_type = ''
                    print("Agent", a.unique_id, "of", j, "move times", "with destination POI of", dest_type)
                    dest_types.append(dest_type)
                    self.grid.move_agent(a, (x, y))
                    a_type = dest_types[-1]

    def step(self):
        self.schedule.step()
