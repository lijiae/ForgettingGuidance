import os
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_data_name(image_dir):
    names=os.listdir(image_dir)
    return names
    

class MaginBrush():
    def __init__(self,config_path,image_dir):
        self.image_names=load_data_name(image_dir)
        self.image_dir=dir
        self.prompt_list=read_json_file(config_path)

    def get_selfcaption(self,name):
        return self.prompt_list[name][name+"-input.png"]
    
    def get_prompt(self,name):
        return self.prompt_list[name][name+"-output1.png"]
