""""""
"""
   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.



Small code to rename embedding keys in state dicts of earlier models (to avoid retraining but allow use of redefined model)
"""
import torch
from collections import OrderedDict
#%%
def rename_emb_key(filename):

    checkpoint = torch.load(filename)
    model_state_dict_new = OrderedDict()
    checkpoint_new = OrderedDict()
    
    for key, value in checkpoint['model_state_dict'].items():
        new_key = key
        if key == "emb_id.weight":
            new_key = "emb.0.weight"        
        if key == "emb.weight":
            new_key = "emb.0.weight" 
        
        model_state_dict_new[new_key] = value
    
    for key, value in checkpoint.items():
        value_new = value
        if key == 'model_state_dict':
            value_new = model_state_dict_new
        
        checkpoint_new[key] = value_new
    
    torch.save(checkpoint_new, filename)
        
dataset_name = 'uci_traffic'

experiment_dir = 'experiments/'+dataset_name
algorithm = 'transformer_conv'
seeds = 10
for seed in range(seeds):
    filename = f'{experiment_dir}/{algorithm}/{algorithm}_seed={seed}'   
    rename_emb_key(filename)