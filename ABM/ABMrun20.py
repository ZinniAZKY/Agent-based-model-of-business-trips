# attention: using ABM30 now
from ABM20 import BusinessModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

starttime = time.time()
admin_grids = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TokyoMesh/TokyoMesh.csv')
agent_freq = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TripTimesFreq/TripTimesFreq.csv')
ori_type_prob_df = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/OriTypeProb/OriTypeProb.csv')
poi_df = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TokyoBusinessPOI/TokyoBusinessPOI.csv')
motif_prob = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/MotifProb/MotifProb.csv')
distance_prob = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/DistanceProb/DistanceProb.csv')
time_prob = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/OriTimeProb/OriTimeProb.csv')

model = BusinessModel(100, admin_grids, agent_freq, ori_type_prob_df, poi_df, motif_prob, distance_prob, time_prob)
for i in range(1):
    model.step()
agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
agent_counts = np.transpose(agent_counts)
plt.imshow(agent_counts, interpolation="nearest", origin="lower")
plt.colorbar()
endtime = time.time()
print(endtime - starttime)
print('finished')
