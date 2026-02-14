from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import numpy as np
import pandas as pd

'''
value_dims = ['Power Distance', 'Individualism', 'Uncertainty Avoidance', 'Masculinity',  'Long-term Orientation', 'Indulgence', 'Corruption', 'Economic Vals', 'Ethical Vals', 'Migration', 'Political Cul', 'Political Int', 'Science', 'Feminist',]
value_dims += ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
              # 'Security', 'Social Capital', 'Social Vals',]
data_random = np.random.rand(1000, 30)
data_random[:, -1] = data_random[:, -2]
data_random[-1,-1] = 0

value_dims = ['Religiosity', 'Neutrality', 'Fairness', 'Skepticism', 'Societal.Tranquility']
five_dimensional_data = pd.read_excel('pone.0245231.s004.xlsx')
five_dimensional_data = five_dimensional_data[value_dims]
#convert pd data to np array
data = five_dimensional_data.to_numpy()
'''

DEFIANCE_DIMS = ["I_AUTHORITY", "I_NATIONALISM", "I_DEVOUT", "DEFIANCE"]
DISBELIEF_DIMS = ["I_RELIGIMP", "I_RELIGBEL", "I_RELIGPRAC", "DISBELIEF"]
RELATIVISM_DIMS = ["I_NORM1", "I_NORM2", "I_NORM3", "RELATIVISM"]
SCEPTICISM_DIMS = ["I_TRUSTARMY", "I_TRUSTPOLICE", "I_TRUSTCOURTS", "SCEPTICISM"]
SEC_DIMS = ["SACSECVAL"]#,"SECVALWGT"]
AUTONOMY_DIMS = ["I_INDEP", "I_IMAGIN", "I_NONOBED", "AUTONOMY"]
EQUALITY_DIMS = ["I_WOMJOB", "I_WOMPOL", "I_WOMEDU", "EQUALITY"]
CHOICE_DIMS = ["I_HOMOLIB", "I_ABORTLIB", "I_DIVORLIB", "CHOICE"]
VOICE_DIMS = ["I_VOICE1", "I_VOICE2", "I_VOI2_00", "VOICE"]
EMA_DIMS = [ "RESEMAVAL"]#, "RESEMAVALWGT"]

dim_sections = [DEFIANCE_DIMS, DISBELIEF_DIMS, RELATIVISM_DIMS, SCEPTICISM_DIMS, SEC_DIMS, AUTONOMY_DIMS, EQUALITY_DIMS, CHOICE_DIMS, VOICE_DIMS, EMA_DIMS]
ALL_ABSTRACT_DIMS = [dim_section[-1] for dim_section in dim_sections]
ALL_DIMS = DEFIANCE_DIMS + DISBELIEF_DIMS + RELATIVISM_DIMS + SCEPTICISM_DIMS + SEC_DIMS + AUTONOMY_DIMS + EQUALITY_DIMS + CHOICE_DIMS + EMA_DIMS #+ VOICE_DIMS

print(ALL_DIMS)
import pdb; pdb.set_trace()


all_dimensional_data = pd.read_csv('WVS_Cross-National_Wave_7_csv_v6_0.csv')
all_dimensional_data = all_dimensional_data[ALL_DIMS]
#import pdb; pdb.set_trace()
#convert pd data to np array
data = all_dimensional_data.to_numpy()
print(data.shape)
data = data[~np.isnan(data).any(axis=1)]
print(data.shape)
#data = np.unique(data, axis=0)
#print(data.shape)
rank = np.linalg.matrix_rank(data)
print(rank)


# default parameters
# cg = pc(data[100:], depth=4)
# # visualization using pydot
# cg.draw_pydot_graph()
# # or save the graph
# pyd = GraphUtils.to_pydot(cg.G)
# pyd.write_png('simple_test.png')

# default parameters
#data = data[]
noise = np.random.normal(0, 0.01, data.shape)  # 0 is the mean of the normal distribution you are choosing from, and 0.01 is the standard deviation of this distribution.
data = data + noise



g, edges = fci(data)
# visualization
from causallearn.utils.GraphUtils import GraphUtils
pdy = GraphUtils.to_pydot(g, labels=ALL_DIMS)
pdy.write_png('simple_test.png')