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
        
        self.index = 1

        if not os.path.exists(database_path):
            os.mkdir(database_path)
        
        #if not os.path.exists(self.scenarios_filename):
        df = pd.DataFrame(columns=['index','topicTitle','topicImg', 'scenarioDescription', "statement"])
        df.to_csv(self.scenarios_filename, index=False)
        #df.to_csv(self.scenarios_filename, index_label='Index')
        #self.scenarios.to_csv(self.scenarios_filename, index=False)
    
        # if not os.path.exists(self.instance_filename):
        #     self.instances = pd.DataFrame(columns=['Scenario ID', 'Instance Content'])
        #     self.instances.to_csv(self.instance_filename, index=False)
        

    def save_new(self, scenario_content, instances):

        #new_row_df = pd.DataFrame([[scenario_content]*len(instances), instances], columns=['Scenario Content', 'Instance'])
        new_row_df = pd.DataFrame({'topicTitle':[self.agent_name+' Agent']*len(instances), 'topicImg':[None]*len(instances), 'scenarioDescription': [scenario_content]*len(instances), "statement": instances})
        start_index = self.index
        end_index = self.index + len(instances)
        new_row_df.index = range(start_index, end_index)
        new_row_df.to_csv(self.scenarios_filename, mode='a', header=False, index=True)

        self.index = end_index

    def csv_to_json(self):
        df = pd.read_csv(self.scenarios_filename)
        df.to_json(self.scenarios_filename[:-3]+'json',orient='records')
