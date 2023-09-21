# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:00:39 2023

@author: jiay
"""
import timeit
import numpy as np
import pandas as pd
from vehicle_choice import MultinomialLogit as mnl

class ModelSpecification(object):
    def __init__(self):
        pass

    v_rescale = {'Base MSRP':0.001, 'Curb weight (lbs)':0.001, 'Shadow (ft2)':0.01,
                'Horsepower (HP)':0.01, 'MPG_Mixed':0.01, 'Range_Mixed':0.01,
                'income_avg':0.001}
    
# =============================================================================
#     v_interactions = [('fuel_HYBRID', 'educ_atleast_College'), ('fuel_ELECTRIC', 'educ_atleast_College'),
#                       ('Shadow (ft2)', 'Household_Size'), ('Shadow (ft2)', 'Household_Size'),
#                       ('fuel_DIESEL', 'income_avg'), ('fuel_HYBRID', 'income_avg'), ('fuel_ELECTRIC', 'income_avg'),
#                       ('Safety_AdapCruise', 'has_children'), ('Safety_Lane', 'has_children'), ('Safety_Brake', 'has_children'), ('Safety_Collision', 'has_children')]
# =============================================================================
    v_interactions = []
    
    attributes = ['log_MSRP/inc', 'log_cost_per_mile/inc'] 
    
    config = {'v_rescale': v_rescale, 'v_interactions': v_interactions, 'attributes': attributes}
 
def f_out(model, res):
      
    out = {z[0] : list(z[1:]) for z in 
           zip(model.X_list, res['Coefficients'].tolist(), np.sqrt(res['Var_Cov']).diagonal())}

    out = pd.DataFrame(out).T
    out = out.rename(columns={0:'Coeff', 1:'Std Err'})
    
    out.loc['sample'] = [len(model.y_grouped), max(set(model.group_size))]
    sample = out.loc['sample']
    out = out.drop('sample')
    out = pd.concat([sample.to_frame().T, out])
    
    return(out)

def model_run(path, file, config):
    start = timeit.default_timer()
    obj = mnl(path, file, config)
    print(obj.X_list)
    bini = np.zeros(len(obj.X_list))
    res = obj.estimation(bini)
    stop = timeit.default_timer()
    print("Runtime: " + str(round((stop-start)/60,2)) + " minutes")    
        
    return f_out(obj, res)

if __name__ == '__main__':
    in_p =r"c:\\users\\jiay\\texas"
    data = "df_YMM_safety_10000_30.parquet"
    run = model_run(in_p, data, ModelSpecification.config)
    
