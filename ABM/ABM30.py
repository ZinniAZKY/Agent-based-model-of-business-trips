import mesa
import numpy as np


# question: since volume between some types of POI is 0, destination choice should be cooperated with original types of POIs
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

    def __init__(self, N, admin_grids, agent_freq, ori_type_prob_df, poi_df, motif_prob, distance_prob):
        self.num_agents = N
        self.admin_grids = admin_grids
        self.agent_freq = agent_freq
        self.ori_type_prob_df = ori_type_prob_df
        self.poi_df = poi_df
        self.motif_prob = motif_prob
        self.read_movement_patterns()
        self.distance_prob = distance_prob
        self.grid = mesa.space.MultiGrid(admin_grids['x'].max() + 1, admin_grids['y'].max() + 1, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.create_agents()

    def calc_cumulative_probs(self):
        # Transform probability of each destination to cumulative probability.\
        self.admin_grids['cum_prob_first'] = self.admin_grids['firstgridprob'].cumsum()
        self.admin_grids['cum_prob_des'] = self.admin_grids['desgridprob'].cumsum()

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

    def assign_original_position(self, agent):
        # Assign an initial position to each agent
        self.admin_grids['firstgridprob'] = self.admin_grids['firstgridprob'] / self.admin_grids[
            'firstgridprob'].sum()
        first_move_row = np.random.choice(self.admin_grids.index, p=self.admin_grids['firstgridprob'])
        ori_x, ori_y = self.admin_grids.loc[first_move_row, ['x', 'y']].values
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

    def generate_distance(self):
        dist_name = np.random.choice(self.distance_prob['distance'], p=self.distance_prob['prob'])
        if dist_name == '2km':
            return 2
        elif dist_name == '2-5km':
            return (2, 5)
        elif dist_name == '5-10km':
            return (5, 10)
        elif dist_name == '10-15km':
            return (10, 15)
        elif dist_name == '15-20km':
            return (15, 20)
        elif dist_name == '20-30km':
            return (20, 30)
        elif dist_name == '30-100km':
            return (30, 100)
        elif dist_name == '100km':
            return 100

    def create_agents(self):
        # read porprotion of each POI.
        type_prob = self.read_type_prob()

        for i in range(self.num_agents):
            # assign number of trips and types of original POI to each agent
            trip_times = self.assign_number_of_moves()
            a = BusinessAgent(i, self, trip_times, self.admin_grids)
            a_type = np.random.choice(list(type_prob.keys()), p=list(type_prob.values()))
            print("Agent", a.unique_id, "Original POI type:", a_type, "trip time:", trip_times)
            # define the original position of agent and set them as default position for distance calculation
            ori_x, ori_y = self.assign_original_position(a)
            dist_x, dist_y = ori_x, ori_y

            # destination choice of agents with number of trips less than 7
            if 2 <= trip_times <= 7:
                # xlist and ylist are used record the previous position of an agent
                xlist = [ori_x]
                ylist = [ori_y]
                # dest_types is used record the previous type of POI of a agent
                dest_types = [a_type]
                # select trip patterns of certain trip number, and give a pattern to an agent proportionally
                patternsubset = self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'pattern']
                a_pattern = np.random.choice(patternsubset, p=self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'prob'])

                for j in range(1, trip_times + 1):
                    # select trip distances proportionally and select grids based on the distance
                    # if grids within distance have no POI, reselect distances
                    sum_desgridprob = 0
                    while sum_desgridprob == 0:
                        # assign distance to one trip proportionally
                        dist_range = self.generate_distance()
                        # calculate distance from the original position to each grids
                        self.admin_grids['distance'] = np.sqrt((self.admin_grids['x'] - dist_x) ** 2 + (self.admin_grids['y'] - dist_y) ** 2)

                        # create boolean mask based on distance range
                        if isinstance(dist_range, int):
                            if dist_range == 2:
                                mask = self.admin_grids['distance'] <= dist_range
                            else:
                                mask = self.admin_grids['distance'] >= dist_range
                        else:
                            mask = (self.admin_grids['distance'] >= dist_range[0]) & (self.admin_grids['distance'] <= dist_range[1])

                        # filter rows based on mask of distance
                        nearby_grids = self.admin_grids.loc[mask]
                        sum_desgridprob = nearby_grids['desgridprob'].sum()

                    nearby_grids['desgridprob'] = nearby_grids['desgridprob'] / sum_desgridprob

                    # agents only move to new grids will not select destinations that have been selected.
                    # (can be improved because agents may move to the same grids but different type of POIs)
                    if all(a_pattern[k] != a_pattern[j] for k in range(j)):
                        des_move_row = np.random.choice(nearby_grids.index,
                                                        p=nearby_grids['desgridprob'])
                        x, y = nearby_grids.loc[des_move_row, ['x', 'y']].values

                        while any((x == x_prev and y == y_prev) for x_prev, y_prev in zip(xlist, ylist)):
                            des_move_row = np.random.choice(nearby_grids.index,
                                                            p=nearby_grids['desgridprob'])
                            x, y = nearby_grids.loc[des_move_row, ['x', 'y']].values

                        # type of destination POI will be assigned based on original type of POI and location
                        # new type of destination POI will be used as original type of POI for next trip
                        # position of dist_x and dist_y will also be updated
                        dest_type = self.generate_dest_type(a_type, x, y)
                        dest_types.append(dest_type)
                        self.grid.move_agent(a, (x, y))
                        xlist.append(x)
                        ylist.append(y)
                        dist_x, dist_y = x, y
                        a_type = dest_types[-1]
                    else:
                        # check all previous grids and move agents to the same grids
                        # the type of POI are also the same with previous one (can be improved to reselect type of POIs)
                        for k, element in enumerate(a_pattern[:j]):
                            if element == a_pattern[j]:
                                prev_index = k
                                dest_type = dest_types[prev_index]
                                dest_types.append(dest_type)
                                self.grid.move_agent(a, (xlist[prev_index], ylist[prev_index]))
                                xlist.append(xlist[prev_index])
                                ylist.append(ylist[prev_index])
                                dist_x, dist_y = x, y
                                a_type = dest_types[-1]
                                break
            # Destination choice of agents with number of trips larger than 7
            else:
                dest_types = [a_type]

                for j in range(1, trip_times + 1):
                    sum_desgridprob = 0

                    while sum_desgridprob == 0:
                        dist_range = self.generate_distance()
                        self.admin_grids['distance'] = np.sqrt((self.admin_grids['x'] - dist_x) ** 2 + (self.admin_grids['y'] - dist_y) ** 2)
                        if isinstance(dist_range, int):
                            if dist_range == 2:
                                mask = self.admin_grids['distance'] <= dist_range
                            else:
                                mask = self.admin_grids['distance'] >= dist_range
                        else:
                            mask = (self.admin_grids['distance'] >= dist_range[0]) & (self.admin_grids['distance'] <= dist_range[1])

                        nearby_grids = self.admin_grids.loc[mask]
                        sum_desgridprob = nearby_grids['desgridprob'].sum()

                    nearby_grids['desgridprob'] = nearby_grids['desgridprob'] / sum_desgridprob
                    des_move_row = np.random.choice(nearby_grids.index, p=nearby_grids['desgridprob'])
                    x, y = nearby_grids.loc[des_move_row, ['x', 'y']].values
                    dest_type = self.generate_dest_type(a_type, x, y)
                    dest_types.append(dest_type)
                    self.grid.move_agent(a, (x, y))
                    dist_x, dist_y = x, y
                    a_type = dest_types[-1]

    def step(self):
        self.schedule.step()
