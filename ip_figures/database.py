from flask import Flask, jsonify, request
import json
import os
import pandas as pd

app = Flask(__name__)

class JSONDatabase:
    def __init__(self):
        ...
        #self.db_file = db_file
        #self.db_file = 'interaction_database//Donald Trump.json'
        #self.data = self._load_database()

    def _load_database(self):
        if not os.path.exists(self.db_file):
            return {}
        with open(self.db_file, 'r') as f:
            return json.load(f)

    def read(self, key=None, index=None):
        if key is None:
            return {}
        json_file_path = os.path.join("interaction_database", key+".json")
        if not os.path.exists(json_file_path):
            return{}
        with open(json_file_path, 'r') as f:
            pddj = pd.read_json(json_file_path)
            filtered_data = pddj[pddj['index'] == index]
            if filtered_data.empty:
                return {}
            else:
                return filtered_data.iloc[0].to_json()
            return json.load(f)
        
        if key is None:
            return self.data
        else:
            return self.data.get(key)

    # def save(self):
    #     with open(self.db_file, 'w') as f:
    #         json.dump(self.data, f, indent=4)

    @app.route('/read', methods=['GET'])
    def read_api():
        key = request.args.get('key')
        index = int(request.args.get('index'))
        #print(key)
        return jsonify(JSONDatabase().read(key, index))

    # @app.route('/update/<string:key>', methods=['POST'])
    # def update_api(key):
    #     value = request.form.get('value')
    #     JSONDatabase('data.json').update_data(key, value)
    #     return jsonify({'status': 'success'})

'''
if __name__ == '__main__':
    JSONDatabase('data.json').save()  # 初始化数据库
    app.run(debug=True)
'''