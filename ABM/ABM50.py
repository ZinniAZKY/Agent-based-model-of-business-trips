import mesa
import warnings
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
import concurrent.futures
import traceback
import math
from keras.models import load_model

warnings.simplefilter(action='ignore', category=FutureWarning)


class BusinessAgent(mesa.Agent):
    """Define the act and condition of business agents."""

    def __init__(self, unique_id, model, move_times, admin_grids):
        super().__init__(unique_id, model)
        self.admin_grids = admin_grids
        self.move_times = move_times

    def move(self):
        for i in range(1, self.move_times + 1):
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


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in kilometers


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

    def __init__(self, N, admin_grids, area_feature, agent_freq, ori_type_prob_df, poi_df, startpoi_df, des_type, motif_prob, distance_prob, time_prob):
        # self.model = load_model(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\DLGModel.h5', compile=False)
        self.all_agents_history = None
        self.num_agents = N
        self.admin_grids = admin_grids
        self.area_feature = area_feature
        self.agent_freq = agent_freq
        self.ori_type_prob_df = ori_type_prob_df
        self.poi_df = poi_df
        self.startpoi_df = startpoi_df
        self.des_type = des_type
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
        self.motif_prob['pattern'] = self.motif_prob['pattern'].str.split('-').apply(
            lambda lst: [str(elem) for elem in lst])

    def assign_number_of_moves(self):
        # assign the number of movements to each agent according to probability
        return np.random.choice(self.agent_freq['times'], p=self.agent_freq['frequency'])

    def assign_original_position(self, agent):
        # assign an original position to each agent
        oripoi_exist = False
        while not oripoi_exist:
            normalized_probs = self.admin_grids['firstgridprob'] / self.admin_grids['firstgridprob'].sum()
            first_move_row = np.random.choice(self.admin_grids.index, p=normalized_probs)
            ori_x, ori_y = self.admin_grids.loc[first_move_row, ['x', 'y']].values
            oriloc_subset = self.startpoi_df[(self.startpoi_df['x'] == ori_x) & (self.startpoi_df['y'] == ori_y)]
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
        ori_type = a_type
        print("Agent", a.unique_id, "Original POI type:", a_type, "trip time:", trip_times)
        # define the original position of agent and set them as default position for distance calculation
        ori_x, ori_y, dist_x, dist_y = self.assign_original_position(a)
        oriloc_poi_subset = self.startpoi_df[(self.startpoi_df['x'] == ori_x) & (self.startpoi_df['y'] == ori_y)]
        ori_poi_x, ori_poi_y = oriloc_poi_subset.sample()[['POI_x', 'POI_y']].values[0]

        return trip_times, a_type, ori_type, a, ori_x, ori_y, dist_x, dist_y, ori_poi_x, ori_poi_y

    def assign_oritime(self, ori_x, ori_y):
        oritime_subset = self.time_prob[(self.time_prob['x'] == ori_x) & (self.time_prob['y'] == ori_y)]
        oritime = np.random.choice(oritime_subset['timestamp'], p=oritime_subset['ODvol'].values)
        if '～' in oritime:
            start_time_str, end_time_str = oritime.split('～')

        else:
            start_time_str, end_time_str = '', oritime

        # Convert the start and end times to datetime objects
        desired_date = datetime.strptime('2012-07-15', '%Y-%m-%d').date()
        start_time = datetime.combine(desired_date, datetime.strptime(start_time_str,
                                                                      '%H:%M').time()) if start_time_str else datetime(
            year=2022, month=1, day=1, hour=0, minute=0)
        end_time = datetime.combine(desired_date,
                                    datetime.strptime(end_time_str, '%H:%M').time()) if end_time_str else datetime(
            year=2022, month=1, day=1, hour=23, minute=59)

        # Generate a random time between the start and end times
        time_diff = end_time - start_time
        random_time = start_time + timedelta(minutes=random.randint(0, time_diff.seconds // 60))
        return random_time

    def predict_destination_DL(self, dist_x, dist_y):
        # Get the AdminID for the origin grid
        origin_admin_id = self.admin_grids.loc[(self.admin_grids['x'] == dist_x) & (self.admin_grids['y'] == dist_y), 'adminid'].values[0]

        # Get the features of the origin admin area
        origin_features = self.area_feature.loc[self.area_feature['adminid'] == origin_admin_id, 'LU100':'y_area']

        # Assuming origin_features is a pandas DataFrame with your features
        origin_df = pd.DataFrame([origin_features.values[0]] * 53, columns=origin_features.columns)
        origin_df.columns = [f"{col}_ori" for col in origin_df.columns]

        # Get the features of each potential destination area
        dest_df = self.area_feature.loc[:, 'LU100':'y_area'].reset_index(drop=True)
        dest_df.columns = [f"{col}_des" for col in dest_df.columns]

        # Add a column for the destination area ID
        dest_df['y'] = self.area_feature['adminid']

        # Concatenate the origin and destination dataframes
        input_df = pd.concat([origin_df, dest_df], axis=1)

        # calculate the haversine distance
        input_df['distance'] = input_df.apply(
            lambda row: haversine_distance(row['x_area_ori'], row['y_area_ori'], row['x_area_des'], row['y_area_des']),
            axis=1)

        # normalize the distance column
        max_distance = 73.67221787
        min_distance = 0.4108906248
        input_df['distance'] = input_df['distance'].clip(min_distance, max_distance)
        input_df['distance'] = (input_df['distance'] - min_distance) / (max_distance - min_distance)

        # list of columns to drop
        drop_columns = ["x_area_ori", "y_area_ori", "x_area_des", "y_area_des", "y"]

        # Drop these columns to get only the feature columns
        model_features = input_df.drop(columns=drop_columns)

        # Convert the features into the format that the model expects (numpy array)
        model_input = model_features.values

        # Use the model to predict the probabilities
        dlmodel = load_model(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\DLGModel.h5', compile=False)
        probabilities = dlmodel.predict(model_input, verbose=0)

        # Convert the numpy array to a pandas DataFrame
        prob_df = pd.DataFrame(data=probabilities, columns=self.area_feature['adminid'].values)

        # Use the max value or the max sum up value of a column to decide destination.
        des_area_prob = prob_df.sum(axis=0) / prob_df.sum(axis=0).sum()
        final_destination = np.random.choice(des_area_prob.index, p=des_area_prob.values)
        # final_destination = prob_df.sum(axis=0).idxmax()
        # final_destination = prob_df.max(axis=0).idxmax()

        return final_destination

    def assign_dest_type(self, ori_type, dist_x, dist_y, xlist=None, ylist=None):
        final_destination = self.predict_destination_DL(dist_x, dist_y)
        # print(final_destination)
        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            nearby_grids = self.filter_buffer_grids(dist_x, dist_y, final_destination)
            # For distance larger than 100km.
            if nearby_grids is None:
                return None, None, None, None, None

            subset_types = self.des_type[self.des_type['ori_type'] == ori_type]['subsettype'].values[0].split(', ')
            poi_subset = self.poi_df[self.poi_df['DesType'].isin(subset_types)]

            nearby_grids_xy = nearby_grids[['x', 'y']]
            poi_subset = poi_subset.merge(nearby_grids_xy, on=['x', 'y'], how='inner')

            if poi_subset.empty:
                attempt += 1
                continue

            selected_poi = poi_subset.sample()
            x, y = selected_poi[['x', 'y']].values[0]

            max_inner_attempts = 100
            inner_attempt = 0

            if xlist is not None and ylist is not None:
                while any((x == x_prev and y == y_prev) for x_prev, y_prev in
                          zip(xlist, ylist)) and inner_attempt < max_inner_attempts:
                    selected_poi = poi_subset.sample()
                    x, y = selected_poi[['x', 'y']].values[0]
                    inner_attempt += 1

                if inner_attempt == max_inner_attempts:
                    print("Too many attempts in the inner loop of assign_dest_type")
                    return None, None, None, None, None

            dest_type = selected_poi['DesType'].values[0]
            poi_x, poi_y = selected_poi[['POI_x', 'POI_y']].values[0]

            return dest_type, x, y, poi_x, poi_y

        print("Infinite loop in assign_dest_type")
        return None, None, None, None, None

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

    # def filter_buffer_grids(self, dist_x, dist_y):
    #     max_attempts = 5000
    #     attempt = 0
    #
    #     while attempt < max_attempts:
    #         # assign distance to one trip, if there are no available grids, reselect different distances.
    #         dist_range = self.assign_distance()
    #         # calculate distance from the original position to each grid
    #         self.admin_grids['distance'] = np.sqrt(
    #             (self.admin_grids['x'] - dist_x) ** 2 + (self.admin_grids['y'] - dist_y) ** 2)
    #
    #         # create boolean mask based on distance range
    #         if dist_range == 2:
    #             mask = self.admin_grids['distance'] <= dist_range
    #         elif dist_range == 100:
    #             return None
    #         else:
    #             mask = (self.admin_grids['distance'] >= dist_range[0]) & (self.admin_grids['distance'] <= dist_range[1])
    #
    #         # filter rows based on mask of distance
    #         nearby_grids = self.admin_grids.loc[mask]
    #
    #         if not nearby_grids.empty:
    #             break
    #
    #         # sum_desgridprob = nearby_grids['desgridprob'].sum()
    #         attempt += 1
    #
    #     if attempt == max_attempts:
    #         print("Infinite loop in filter_buffer_grids")
    #         return None
    #
    #     nearby_grids = nearby_grids.copy()
    #     # nearby_grids.loc[:, 'desgridprob'] = nearby_grids['desgridprob'] / sum_desgridprob
    #     return nearby_grids

    def filter_buffer_grids(self, dist_x, dist_y, final_destination):
        max_attempts = 100
        attempt = 0

        while attempt < max_attempts:
            # assign distance to one trip, if there are no available grids, reselect different distances.
            dist_range = self.assign_distance()
            # calculate distance from the original position to each grid
            self.admin_grids['distance'] = np.sqrt(
                (self.admin_grids['x'] - dist_x) ** 2 + (self.admin_grids['y'] - dist_y) ** 2)

            # create boolean mask based on distance range
            if dist_range == 2:
                mask = self.admin_grids['distance'] <= dist_range
            elif dist_range == 100:
                return None
            else:
                mask = (self.admin_grids['distance'] >= dist_range[0]) & (self.admin_grids['distance'] <= dist_range[1])

            # Add an extra condition to the mask to filter out grids that do not belong to the final destination adminid
            mask = mask & (self.admin_grids['adminid'] == final_destination)

            # filter rows based on mask of distance and adminid
            nearby_grids = self.admin_grids.loc[mask]

            if not nearby_grids.empty:
                break

            # sum_desgridprob = nearby_grids['desgridprob'].sum()
            attempt += 1

        if attempt == max_attempts:
            print("Infinite loop in filter_buffer_grids")
            return None

        nearby_grids = nearby_grids.copy()
        # nearby_grids.loc[:, 'desgridprob'] = nearby_grids['desgridprob'] / sum_desgridprob
        return nearby_grids

    def assign_pattern(self, trip_times):
        # select trip patterns of certain trip number, and give a pattern to an agent proportionally
        pattern_subset = self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'pattern']
        a_pattern = np.random.choice(pattern_subset,
                                     p=self.motif_prob.loc[self.motif_prob['tripnum'] == trip_times, 'prob'])
        return a_pattern

    def process_agent(self, i, type_prob):

        average_speed = 30  # in km/h
        stay_time_mean = 45
        stay_time_sd = 7.5
        stay_time_lower, stay_time_upper = 30, 60

        trip_times, a_type, ori_type, a, ori_x, ori_y, dist_x, dist_y, ori_poi_x, ori_poi_y = self.initialize_agent(i, type_prob)
        random_time = self.assign_oritime(ori_x, ori_y)
        self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, ori_x, ori_y, ori_poi_x, ori_poi_y,
                                                                           a_type, random_time]

        # destination choice of agents with number of trips less than 7
        if 2 <= trip_times <= 7:
            xlist, ylist, dest_types, poi_xlist, poi_ylist = [ori_x], [ori_y], [a_type], [ori_poi_x], [ori_poi_y]
            a_pattern = self.assign_pattern(trip_times)

            for j in range(1, trip_times + 1):
                # agents only move to new grids will not select destinations that have been selected.
                # (can be improved because agents may move to the same grids but different type of POIs)

                if all(a_pattern[k] != a_pattern[j] for k in range(j)):
                    dest_type, x, y, poi_x, poi_y = self.assign_dest_type(ori_type, dist_x, dist_y, xlist, ylist)
                    if dest_type is None:
                        self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, None, None, None, None,
                                                                                           None, '']
                        return self.all_agents_history

                    # 将计算时间复制到其他分支
                    distance = haversine_distance(ori_poi_x, ori_poi_y, poi_x, poi_y)
                    travel_time_hours = distance / average_speed
                    travel_time_minutes = travel_time_hours * 60
                    stay_times = np.random.normal(stay_time_mean, stay_time_sd, size=1)
                    stay_times = np.clip(stay_times, stay_time_lower, stay_time_upper)[0]
                    departure_time = random_time + timedelta(minutes=travel_time_minutes) + timedelta(minutes=stay_times)
                    random_time = departure_time
                    ori_poi_x, ori_poi_y = poi_x, poi_y

                    dest_types.append(dest_type)
                    xlist.append(x)
                    ylist.append(y)
                    poi_xlist.append(poi_x)
                    poi_ylist.append(poi_y)
                    dist_x, dist_y = x, y
                    a_type = dest_types[-1]
                    self.grid.move_agent(a, (x, y))

                    self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, x, y, poi_x, poi_y, a_type, departure_time]
                else:
                    # check all previous grids and move agents to the same grids
                    # the type of POI are also the same with previous one (can be improved to reselect type of POIs)
                    for k, element in enumerate(a_pattern[:j]):
                        if element == a_pattern[j]:
                            dest_type = dest_types[k]
                            dest_types.append(dest_type)
                            poi_x, poi_y = poi_xlist[k], poi_ylist[k]
                            x, y = xlist[k], ylist[k]

                            distance = haversine_distance(ori_poi_x, ori_poi_y, poi_x, poi_y)
                            travel_time_hours = distance / average_speed
                            travel_time_minutes = travel_time_hours * 60
                            stay_times = np.random.normal(stay_time_mean, stay_time_sd, size=1)
                            stay_times = np.clip(stay_times, stay_time_lower, stay_time_upper)[0]
                            departure_time = random_time + timedelta(minutes=travel_time_minutes) + timedelta(
                                minutes=stay_times)
                            random_time = departure_time
                            ori_poi_x, ori_poi_y = poi_x, poi_y

                            poi_xlist.append(poi_x)
                            poi_ylist.append(poi_y)
                            xlist.append(x)
                            ylist.append(y)
                            dist_x, dist_y = x, y
                            a_type = dest_types[-1]
                            self.grid.move_agent(a, (x, y))
                            self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, x, y, poi_x, poi_y,
                                                                                               a_type, departure_time]
                            break
        # Destination choice of agents with number of trips larger than 7
        else:
            dest_types = [a_type]
            for j in range(1, trip_times + 1):
                dest_type, x, y, poi_x, poi_y = self.assign_dest_type(ori_type, dist_x, dist_y)
                if dest_type is None:
                    self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, None, None, None, None, None,
                                                                                       '']
                    return self.all_agents_history

                distance = haversine_distance(ori_poi_x, ori_poi_y, poi_x, poi_y)
                travel_time_hours = distance / average_speed
                travel_time_minutes = travel_time_hours * 60
                stay_times = np.random.normal(stay_time_mean, stay_time_sd, size=1)
                stay_times = np.clip(stay_times, stay_time_lower, stay_time_upper)[0]
                departure_time = random_time + timedelta(minutes=travel_time_minutes) + timedelta(
                    minutes=stay_times)
                random_time = departure_time
                ori_poi_x, ori_poi_y = poi_x, poi_y

                dest_types.append(dest_type)
                dist_x, dist_y = x, y
                a_type = dest_types[-1]
                self.grid.move_agent(a, (x, y))
                self.all_agents_history.loc[len(self.all_agents_history.index)] = [i, x, y, poi_x, poi_y, a_type, departure_time]
        return self.all_agents_history

    def create_agents(self):
        # read proportion of each POI.
        type_prob = self.read_type_prob()
        self.all_agents_history = pd.DataFrame(
            columns=['agent_id', 'grid_x', 'grid_y', 'geo_x', 'geo_y', 'type', 'oritime'])

        timeout = 600

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(self.num_agents):
                future = executor.submit(self.process_agent, i, type_prob)
                try:
                    result = future.result(timeout)
                    # Update the all_agents_history DataFrame with the result
                    self.all_agents_history = result
                except concurrent.futures.TimeoutError:
                    print(f"Timeout reached for agent {i}, skipping this agent")
                    traceback.print_exc()  # Print the call stack
                    continue

    def step(self):
        self.schedule.step()
