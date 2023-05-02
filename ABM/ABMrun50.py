from ABM50 import BusinessModel
import pandas as pd
import time
from multiprocessing import Pool


def run_business_model(params, model_index, agent_id_offset):
    try:
        # Unpack parameters. Adding new input data need to change parameters in line 10, 11, 25.
        N, admin_grids, area_feature, agent_freq, ori_type_prob_df, poi_df, startpoi_df, des_type, motif_prob, distance_prob, time_prob = params
        model = BusinessModel(N, admin_grids, area_feature, agent_freq, ori_type_prob_df, poi_df, startpoi_df, des_type, motif_prob, distance_prob, time_prob)
        for i in range(1):
            model.step()
            model.all_agents_history['agent_id'] += agent_id_offset
            model.all_agents_history.to_csv(f'C:\\Users\\zhang\\PycharmProjects\\pythonProject\\Doctor Research '
                                            f'4Q\\ABM\\agent_trips_{model_index}.csv', index=False)
            print("create csv file finished")
    except Exception as e:
        print(f"Error in model {model_index}: {e}")
        return None

    return model


def run_models_in_parallel(n_models, n_processes):
    params = [(
        10,
        admin_grids,
        area_feature,
        agent_freq,
        ori_type_prob_df,
        poi_df,
        startpoi_df,
        des_type,
        motif_prob,
        distance_prob,
        time_prob
    ) for _ in range(n_models)]

    with Pool(n_processes) as pool:
        models = pool.starmap(run_business_model, [(params, i, i * 10) for i, params in enumerate(params)])
        print("run_models_in_parallel finished.")
    return models


if __name__ == "__main__":
    starttime = time.time()

    admin_grids = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TokyoMesh/TokyoMesh.csv')
    area_feature = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\AdminFeature/AdminFeature.csv')
    agent_freq = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TripTimesFreq/TripTimesFreq.csv')
    ori_type_prob_df = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\OriTypeProb/OriTypeProb2.csv')
    poi_df = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TokyoBusinessPOI/TokyoBusinessPOI2.csv')
    startpoi_df = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TokyoOfficePOI/TokyoOfficePOI.csv')
    des_type = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\DesTypeList/DesTypeList.csv')
    motif_prob = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\MotifProb/MotifProb.csv')
    distance_prob = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\DistanceProb/DistanceProb.csv')
    time_prob = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\OriTimeProb/OriTimeProb_2.csv')

    n_models = 4
    n_processes = 4
    models = run_models_in_parallel(n_models, n_processes)
    endtime = time.time()
    print(f"Execution time: {endtime - starttime} seconds")
    print('finished')

# 不使用multiprocessing
# def run_business_model(params, model_index, agent_id_offset):
#     try:
#         # Unpack parameters. Adding new input data need to change parameters in line 10, 11, 25.
#         N, admin_grids, area_feature, agent_freq, ori_type_prob_df, poi_df, startpoi_df, des_type, motif_prob, distance_prob, time_prob = params
#         model = BusinessModel(N, admin_grids, area_feature, agent_freq, ori_type_prob_df, poi_df, startpoi_df, des_type, motif_prob, distance_prob, time_prob)
#         for i in range(1):
#             model.step()
#             model.all_agents_history['agent_id'] += agent_id_offset
#             model.all_agents_history.to_csv(f'C:\\Users\\zhang\\PycharmProjects\\pythonProject\\Doctor Research '
#                                             f'4Q\\ABM\\agent_trips_{model_index}.csv', index=False)
#             print("create csv file finished")
#     except Exception as e:
#         print(f"Error in model {model_index}: {e}")
#         return None
#
#     return model
#
#
# def run_models(n_models):
#     params = [(
#         100,
#         admin_grids,
#         area_feature,
#         agent_freq,
#         ori_type_prob_df,
#         poi_df,
#         startpoi_df,
#         des_type,
#         motif_prob,
#         distance_prob,
#         time_prob
#     ) for _ in range(n_models)]
#
#     models = []
#     for i, params in enumerate(params):
#         model = run_business_model(params, i, i * 100)
#         models.append(model)
#         print("Model finished.")
#     return models
#
#
# if __name__ == "__main__":
#     starttime = time.time()
#
#     admin_grids = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TokyoMesh/TokyoMesh.csv')
#     area_feature = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\AdminFeature/AdminFeature.csv')
#     agent_freq = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TripTimesFreq/TripTimesFreq.csv')
#     ori_type_prob_df = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\OriTypeProb/OriTypeProb2.csv')
#     poi_df = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TokyoBusinessPOI/TokyoBusinessPOI2.csv')
#     startpoi_df = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\TokyoOfficePOI/TokyoOfficePOI.csv')
#     des_type = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\DesTypeList/DesTypeList.csv')
#     motif_prob = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\MotifProb/MotifProb.csv')
#     distance_prob = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\DistanceProb/DistanceProb.csv')
#     time_prob = pd.read_csv(r'C:\Users\zhang\PycharmProjects\pythonProject\Doctor Research 4Q\ABM\OriTimeProb/OriTimeProb_2.csv')
#
#     n_models = 1
#     models = run_models(n_models)
#     endtime = time.time()
#     print(f"Execution time: {endtime - starttime} seconds")
#     print('finished')

