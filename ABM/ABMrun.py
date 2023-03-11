from ABMtest import BusinessModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

admin_grids_df = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TokyoMesh/TokyoMesh.csv')
agent_freq = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/TripTimesFreq/TripTimesFreq.csv')
type_prob_df = pd.read_csv('/Users/zhangkunyi/PythonCode/Doctor Research 4Q/ABM/OriTypeProb/OriTypeProb.csv')
model = BusinessModel(1000, admin_grids_df, agent_freq, type_prob_df)
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
print('finished')
