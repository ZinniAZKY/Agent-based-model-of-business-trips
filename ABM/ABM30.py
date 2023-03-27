import mesa
import numpy as np
import warnings
import pandas as pd
from datetime import datetime, timedelta, time
import random

warnings.simplefilter(action='ignore', category=FutureWarning)


class BusinessAgent(mesa.Agent):
    """Define the act and condition of business agents."""

    def __init__(self, unique_id, model, move_times, admin_grids):
        super().__init__(unique_id, model)
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

    def step(self):
        self.move()


class BusinessModel(mesa.Model):
    """Variables that will be used in this class:
        agent: business agents that will travel between POIs.
        a_type: a_type will be used to record types of each POI from the original POI to the last one.
        ori_x, ori_y: coordinates of the 108*54 grids and is used for filtering all kinds of subset.
        dist_x, dist_y: coordinates of the 108*54 grids and is only used for calculating distance.
        ori_poi_x, ori_poi_y: geographical coordinates of the original POI of each agent.
        x, y: temporary coordinates of the 108*54 grids of agent's each destination.
        poi_x, poi_y: temporary geographical coordinates of agent's each destination direved from x, y.
        oriloc_subset, oriloc_poi_subset, oritime_subset, poi_subset, pattern_subset: all the temporary subset."""

    def __init__(self, N, admin_grids, agent_freq, ori_type_prob_df, poi_df, motif_prob, distance_prob, time_prob):
        self.all_agents_history = None
        self.num_agents = N
        self.admin_grids = admin_grids
        self.agent_freq = agent_freq
        self.ori_type_prob_df = ori_type_prob_df
        self.poi_df = poi_df
        self.motif_prob = motif_prob
        self.read_movement_patterns()
        self.distance_prob = distance_prob
        self.time_prob = time_prob
        self.grid = mesa.space.MultiGrid(admin_grids['x'].max() + 1, admin_grids['y'].max() + 1, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.create_agents()

    def read_type_prob(self):
        # read probability of types of origin POIs
        return self.ori_type_prob_df.set_index('Type')['Percentage'].to_dict()

    def read_movement_patterns(self):
        # reads the agent's movement patterns as a list
        self.motif_prob['pattern'] = self.motif_prob['pattern'].str.split('-').apply(lambda lst: [str(elem) for elem in lst])

    def assign_number_of_moves(self):
        # assign the number of movements to each agent according to probability
        return np.random.choice(self.agent_freq['times'], p=self.agent_freq['frequency'])

    def assign_original_position(self, agent, a_type):
        # assign an original position to each agent
        oripoi_exist = False
        while not oripoi_exist:
            self.admin_grids['firstgridprob'] = self.admin_grids['desgridprob'] / self.admin_grids['desgridprob'].sum()
            first_move_row = np.random.choice(self.admin_grids.index, p=self.admin_grids['firstgridprob'])
            ori_x, ori_y = self.admin_grids.loc[first_move_row, ['x', 'y']].values
            oriloc_subset = self.poi_df[
                (self.poi_df['DesType'] == a_type) & (self.poi_df['x'] == ori_x) & (self.poi_df['y'] == ori_y)]
            oripoi_exist = oriloc_subset.shape[0]
        self.grid.place_agent(agent, (ori_x, ori_y))
        # dist_x and dist_y are the same as ori_x, ori_y at the original position
        return ori_x, ori_y, ori_x, ori_y

    def initialize_agent(self, i, type_prob):
        # assign number of trips and types of original POI to each agent
        # use location of POI as location of each agent's start point
        trip_times = self.assign_number_of_moves()
        a = BusinessAgent(i, self, trip_times, self.admin_grids)
        a_type = np.random.choice(list(type_prob.keys()), p=list(type_prob.values()))
        print("Agent", a.unique_id, "Original POI type:", a_type, "trip time:", trip_times)
        # define the original position of agent and set them as default position for distance calculation
        ori_x, ori_y, dist_x, dist_y = self.assign_original_position(a, a_type)
        oriloc_poi_subset = self.poi_df[(self.poi_df['DesType'] == a_type) & (self.poi_df['x'] == ori_x) & (self.poi_df['y'] == ori_y)]
        ori_poi_x, ori_poi_y = oriloc_poi_subset.sample()[['POI_x', 'POI_y']].values[0]

        return trip_times, a_type, a, ori_x, ori_y, dist_x, dist_y, ori_poi_x, ori_poi_y

    def assign_oritime(self, ori_x, ori_y):
        oritime_subset = self.time_prob[(self.time_prob['x'] == ori_x) & (self.time_prob['y'] == ori_y)]
        oritime = np.random.choice(oritime_subset['timestamp'], p=oritime_subset['ODvol'].values)
        if '～' in oritime:
            start_time_str, end_time_str = oritime.split('～')

        else:
            start_time_str, end_time_str = '', oritime

        # Convert the start and end times to datetime objects
        desired_date = datetime.strptime('2012-07-15', '%Y-%m-%d').date()
        start_time = datetime.combine(desired_date, datetime.strptime(start_time_str, '%H:%M').time()) if start_time_str else datetime(year=2022, month=1, day=1, hour=0, minute=0)
        end_time = datetime.combine(desired_date, datetime.strptime(end_time_str, '%H:%M').time()) if end_time_str else datetime(year=2022, month=1, day=1, hour=23, minute=59)

        # Generate a random time between the start and end times
        time_diff = end_time - start_time
        random_time = start_time + timedelta(minutes=random.randint(0, time_diff.seconds // 60))
        return random_time

    def assign_dest_type(self, a_type, dist_x, dist_y, xlist=None, ylist=None):
        # filter types of POI based on location and OD matrix
        # select trip distances proportionally and select grids based on the distance
        # if grids within distance have no POI, reselect distances
        sum_dest_volume = 0
        nearby_grids = self.filter_buffer_grids(dist_x, dist_y)

        while sum_dest_volume == 0:
            des_move_row = np.random.choice(nearby_grids.index, p=nearby_grids['desgridprob'])
            x, y = nearby_grids.loc[des_move_row, ['x', 'y']].values

            if xlist is not None and ylist is not None:
                while any((x == x_prev and y == y_prev) for x_prev, y_prev in zip(xlist, ylist)):
                    des_move_row = np.random.choice(nearby_grids.index, p=nearby_grids['desgridprob'])
                    x, y = nearby_grids.loc[des_move_row, ['x', 'y']].values

            poi_subset = self.poi_df[
                (self.poi_df['OriType'] == a_type) & (self.poi_df['x'] == x) & (self.poi_df['y'] == y)]
            dest_volumes_subset = dict(zip(poi_subset['DesType'], poi_subset['volume']))
            dest_volumes_list = [v for v in dest_volumes_subset.values()]
            sum_dest_volume = sum(dest_volumes_list)

        dest_volumes_norm = np.array(dest_volumes_list) / sum_dest_volume
        dest_type_idx = np.random.choice(len(dest_volumes_list), p=dest_volumes_norm)
        dest_type = list(dest_volumes_subset.keys())[dest_type_idx]
        poi_x, poi_y = poi_subset[poi_subset['DesType'] == dest_type].sample()[['POI_x', 'POI_y']].values[0]

        return dest_type, x, y, poi_x, poi_y

    def assign_distance(self):
        # assign buffer to select grids of each trip
        dist_name = np.random.choice(self.distance_prob['distance'], p=self.distance_prob['prob'])
        dist_ranges = {
            '2km': 2,
            '2-5km': (2, 5),
            '5-10km': (5, 10),
            '10-15km': (10, 15),
            '15-20km': (15, 20),
            '20-30km': (20, 30),
            '30-100km': (30, 100),
            '100km': 100
        }
        return dist_ranges[dist_name]

    def filter_buffer_grids(self, dist_x, dist_y):
        sum_desgridprob = 0

        while sum_desgridprob == 0:
            # assign distance to one trip proportionally
            dist_range = self.assign_distance()
            # calculate distance from the original position to each grids
            self.admin_grids['distance'] = np.sqrt((self.admin_grids['x'] - dist_x) ** 2 + (self.admin_grids['y'] - dist_y) ** 2)

            # create boolean mask based on distance range
            if dist_range == 2:
                mask = self.admin_grids['distance'] <= dist_range
            elif dist_range == 100:
                mask = self.admin_grids['distance'] >= dist_range
            else:
                mask = (self.admin_grids['distance'] >= dist_range[0]) & (self.admin_grids['distance'] <= dist_range[1])

            # filter rows based on mask of distance
            nearby_grids = self.admin_grids.loc[mask]
            sum_desgridprob = nearby_grids['desgridprob'].sum()

        nearby_grids = nearby_grids.copy()
        nearby_grids.loc[:, 'desgridprob'] = nearby_grids['desgridprob'] / sum_desgridprob
        return nearby_grids

    def assign_pattern(self, trip_times):
        # select trip patterns of certain trip number, and give a pattern to an agent proportionally
        pattern_subset = self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'pattern']
        a_pattern = np.random.choice(pattern_subset,
                                     p=self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'prob'])
        return a_pattern

    def create_agents(self):
        # read porprotion of each POI.
        type_prob = self.read_type_prob()
        self.all_agents_history = pd.DataFrame(columns=['agent_id', 'grid_x', 'grid_y', 'geo_x', 'geo_y', 'type', 'oritime'])

        for i in range(self.num_agents):
            trip_times, a_type, a, ori_x, ori_y, dist_x, dist_y, ori_poi_x, ori_poi_y = self.initialize_agent(i, type_prob)
            random_time = self.assign_oritime(ori_x, ori_y)
            self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, ori_x, ori_y, ori_poi_x, ori_poi_y, a_type, random_time]
            # destination choice of agents with number of trips less than 7
            if 2 <= trip_times <= 7:
                xlist, ylist, dest_types, poi_xlist, poi_ylist = [ori_x], [ori_y], [a_type], [ori_poi_x], [ori_poi_y]
                a_pattern = self.assign_pattern(trip_times)

                for j in range(1, trip_times + 1):
                    # agents only move to new grids will not select destinations that have been selected.
                    # (can be improved because agents may move to the same grids but different type of POIs)
                    if all(a_pattern[k] != a_pattern[j] for k in range(j)):
                        dest_type, x, y, poi_x, poi_y = self.assign_dest_type(a_type, dist_x, dist_y, xlist, ylist)
                        dest_types.append(dest_type)
                        xlist.append(x)
                        ylist.append(y)
                        poi_xlist.append(poi_x)
                        poi_ylist.append(poi_y)
                        dist_x, dist_y = x, y
                        a_type = dest_types[-1]
                        self.grid.move_agent(a, (x, y))
                        self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, x, y, poi_x, poi_y, a_type, '']
                    else:
                        # check all previous grids and move agents to the same grids
                        # the type of POI are also the same with previous one (can be improved to reselect type of POIs)
                        for k, element in enumerate(a_pattern[:j]):
                            if element == a_pattern[j]:
                                dest_type = dest_types[k]
                                dest_types.append(dest_type)
                                poi_x, poi_y = poi_xlist[k], poi_ylist[k]
                                poi_xlist.append(poi_x)
                                poi_ylist.append(poi_y)
                                xlist.append(x)
                                ylist.append(y)
                                dist_x, dist_y = x, y
                                a_type = dest_types[-1]
                                self.grid.move_agent(a, (x, y))
                                self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, x, y, poi_x, poi_y, a_type, '']
                                break
            # Destination choice of agents with number of trips larger than 7
            else:
                dest_types = [a_type]

                for j in range(1, trip_times + 1):
                    dest_type, x, y, poi_x, poi_y = self.assign_dest_type(a_type, dist_x, dist_y)
                    dest_types.append(dest_type)
                    dist_x, dist_y = x, y
                    a_type = dest_types[-1]
                    self.grid.move_agent(a, (x, y))
                    self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, x, y, poi_x, poi_y, a_type, '']

        # all_agents_history.to_csv('/Users/zhangkunyi/Downloads/agent_trips.csv', index=False)

    def step(self):
        self.schedule.step()

