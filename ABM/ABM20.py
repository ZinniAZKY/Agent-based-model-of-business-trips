import mesa
import numpy as np
import numba


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

    def __init__(self, N, admin_grids_df, agent_freq, ori_type_prob_df, poi_df, motif_prob, distance_prob):
        self.num_agents = N
        self.admin_grids_df = admin_grids_df
        self.admin_grids = None
        self.load_admin_grids()
        self.agent_freq = agent_freq
        self.ori_type_prob_df = ori_type_prob_df
        self.poi_df = poi_df
        self.motif_prob = motif_prob
        self.distance_prob = distance_prob
        self.grid = mesa.space.MultiGrid(admin_grids_df['x'].max() + 1, admin_grids_df['y'].max() + 1, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.create_agents()

    def load_admin_grids(self):
        # Load the Tokyo administrative area grid
        if self.admin_grids is None:
            self.admin_grids = self.admin_grids_df.copy()

    def calc_cumulative_probs(self):
        # Transform probability of each destination to cumulative probability.\
        self.admin_grids_df['cum_prob_first'] = self.admin_grids_df['firstgridprob'].cumsum()
        self.admin_grids_df['cum_prob_des'] = self.admin_grids_df['desgridprob'].cumsum()

    def read_type_prob(self):
        # Read probability of types of origin POIs
        type_prob = self.ori_type_prob_df.set_index('Type')['Percentage'].to_dict()
        return type_prob

    def read_movement_patterns(self):
        # # Reads the agent's movement patterns as a list
        self.motif_prob['pattern'] = self.motif_prob['pattern'].str.split('-').apply(
            lambda lst: [str(elem) for elem in lst])

    def assign_number_of_moves(self):
        # Assign the number of moves to each agent according to probability.
        return np.random.choice(self.agent_freq['times'], p=self.agent_freq['frequency'])

    def move_agent_to_position(self, position):
        self.grid.move_agent(self, position)

    def assign_initial_position(self, agent):
        # Assign an initial position to each agent
        self.admin_grids_df['firstgridprob'] = self.admin_grids_df['firstgridprob'] / self.admin_grids_df[
            'firstgridprob'].sum()
        first_move_row = np.random.choice(self.admin_grids_df.index, p=self.admin_grids_df['firstgridprob'])
        ori_x, ori_y = self.admin_grids_df.loc[first_move_row, ['x', 'y']].values
        self.grid.place_agent(agent, (ori_x, ori_y))
        return ori_x, ori_y

    def generate_dest_type(self, a_type, x, y):
        poi_subset = self.poi_df[(self.poi_df['OriType'] == a_type) & (self.poi_df['x'] == x) & (self.poi_df['y'] == y)]
        dest_volumes_subset = dict(zip(poi_subset['DesType'], poi_subset['volume']))

        if len(dest_volumes_subset) > 0:
            dest_volumes_list = [v for v in dest_volumes_subset.values()]
            dest_volumes_norm = np.array(dest_volumes_list) / sum(dest_volumes_list)
            dest_type_idx = np.random.choice(len(dest_volumes_list), p=dest_volumes_norm)
            dest_type = list(dest_volumes_subset.keys())[dest_type_idx]
        else:
            dest_type = ''
            print('Destination choice model locates in grids with no POIs.')

        return dest_type

    def create_agents(self):
        # Read POI type probabilities
        type_prob = self.read_type_prob()
        trip_times = self.assign_number_of_moves()

        for i in range(self.num_agents):

            # POI type to assign initial position to each agent
            a = BusinessAgent(i, self, trip_times, self.admin_grids)
            a_type = np.random.choice(list(type_prob.keys()), p=list(type_prob.values()))
            print("Agent", a.unique_id, "Original POI type:", a_type)
            a.type = a_type

            ori_x, ori_y = self.assign_initial_position(a)

            if 2 <= trip_times <= 7:
                xlist = [ori_x]
                ylist = [ori_y]
                dest_types = [a_type]
                patternsubset = self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'pattern']
                a_pattern = np.random.choice(patternsubset,
                                             p=self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'prob'])
                for j in range(1, trip_times + 1):

                    if all(a_pattern[k] != a_pattern[j] for k in range(j)):
                        des_move_row = np.random.choice(self.admin_grids_df.index,
                                                        p=self.admin_grids_df['desgridprob'])
                        x, y = self.admin_grids_df.loc[des_move_row, ['x', 'y']].values

                        while any((x == xprev and y == yprev) for xprev, yprev in zip(xlist, ylist)):
                            des_move_row = np.random.choice(self.admin_grids_df.index,
                                                            p=self.admin_grids_df['desgridprob'])
                            x, y = self.admin_grids_df.loc[des_move_row, ['x', 'y']].values

                        dest_type = self.generate_dest_type(a_type, x, y)
                        dest_types.append(dest_type)
                        self.grid.move_agent(a, (x, y))
                        xlist.append(x)
                        ylist.append(y)
                        a_type = dest_types[-1]
                    else:
                        for k, element in enumerate(a_pattern[:j]):
                            if element == a_pattern[j]:
                                prev_index = k
                                dest_type = dest_types[prev_index]
                                dest_types.append(dest_type)
                                self.grid.move_agent(a, (xlist[prev_index], ylist[prev_index]))
                                xlist.append(xlist[prev_index])
                                ylist.append(ylist[prev_index])
                                a_type = dest_types[-1]
                                break
            else:
                dest_types = [a_type]
                # Use numpy.random.choice to select a random row directly
                for j in range(1, trip_times + 1):
                    des_move_row = np.random.choice(self.admin_grids_df.index, p=self.admin_grids_df['desgridprob'])
                    x, y = self.admin_grids_df.loc[des_move_row, ['x', 'y']].values
                    dest_type = self.generate_dest_type(a_type, x, y)
                    dest_types.append(dest_type)
                    self.grid.move_agent(a, (x, y))
                    a_type = dest_types[-1]

    def step(self):
        self.schedule.step()
