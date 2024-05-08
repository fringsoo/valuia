import json
from datetime import datetime
import os
import pandas as pd

class InteractionCollector:
    def __init__(self, agent_name, database_path):
        self.agent_name = agent_name
        self.scenarios_filename = database_path + f"//{agent_name}.csv"
        #self.scenarios_filename = database_path + f"//{agent_name}_scenarios.csv"
        #self.instance_filename = database_path + f"//{agent_name}_instances.csv"
        
        if not os.path.exists(database_path):
            os.mkdir(database_path)
        
        if not os.path.exists(self.scenarios_filename):
            df = pd.DataFrame(columns=['Scenario Content', 'Instance'])
            df.to_csv(self.scenarios_filename, index=False)
            #df.to_csv(self.scenarios_filename, index_label='Index')
            #self.scenarios.to_csv(self.scenarios_filename, index=False)
        
        # if not os.path.exists(self.instance_filename):
        #     self.instances = pd.DataFrame(columns=['Scenario ID', 'Instance Content'])
        #     self.instances.to_csv(self.instance_filename, index=False)
        

    def save_new(self, scenario_content, instances):

        #new_row_df = pd.DataFrame([[scenario_content]*len(instances), instances], columns=['Scenario Content', 'Instance'])
        new_row_df = pd.DataFrame({'Scenario Content': [scenario_content]*len(instances), "Instance": instances})
        #new_row_df = pd.DataFrame([[scenario_content, 'bar'], ['hello', 'world']])
        new_row_df.to_csv(self.scenarios_filename, mode='a', header=False, index=False)
    