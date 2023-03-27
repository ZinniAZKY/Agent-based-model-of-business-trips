from ABM30 import BusinessModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from multiprocessing import Pool


def run_business_model(params, model_index):
    # Unpack parameters
    N, admin_grids, agent_freq, ori_type_prob_df, poi_df, motif_prob, distance_prob, time_prob= params

    model = BusinessModel(N, admin_grids, agent_freq, ori_type_prob_df, poi_df, motif_prob, distance_prob, time_prob)
    for i in range(1):
        model.step()
        model.all_agents_history.to_csv(f'/Users/zhangkunyi/Downloads/agent_trips_{model_index}.csv', index=False)

    return model


def run_models_in_parallel(n_models, n_processes):
    params = [(
        25,
        admin_grids,
        agent_freq,
        ori_type_prob_df,
        poi_df,
        motif_prob,
        distance_prob,
        time_prob
    ) for _ in range(n_models)]

    with Pool(n_processes) as pool:
        models = pool.starmap(run_business_model, [(params, i) for i, params in enumerate(params)])

    return models


if __name__ == "__main__":
    starttime = time.time()

    admin_grids = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TokyoMesh/TokyoMesh.csv')
    agent_freq = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TripTimesFreq/TripTimesFreq.csv')
    ori_type_prob_df = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/OriTypeProb/OriTypeProb.csv')
    poi_df = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TokyoBusinessPOI/TokyoBusinessPOI.csv')
    motif_prob = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/MotifProb/MotifProb.csv')
    distance_prob = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/DistanceProb/DistanceProb.csv')
    # add
    time_prob = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/OriTimeProb/OriTimeProb.csv')

    n_models = 4
    n_processes = 4
    models = run_models_in_parallel(n_models, n_processes)
    endtime = time.time()
    print(f"Execution time: {endtime - starttime} seconds")
    print('finished')

    # model = BusinessModel(1000, admin_grids, agent_freq, ori_type_prob_df, poi_df, motif_prob, distance_prob)
    # for i in range(1):
    #     model.step()
    # agent_counts = np.zeros((model.grid.width, model.grid.height))
    # for cell in model.grid.coord_iter():
    #     cell_content, x, y = cell
    #     agent_count = len(cell_content)
    #     agent_counts[x][y] = agent_count
    # agent_counts = np.transpose(agent_counts)
    # plt.imshow(agent_counts, interpolation="nearest", origin="lower")
    # plt.colorbar()
    # endtime = time.time()
    # print(endtime - starttime)

