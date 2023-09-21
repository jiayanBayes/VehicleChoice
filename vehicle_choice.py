# -*- coding: utf-8 -*-
"""
@author: jiay
"""
import os
import random
import numpy as np
import pandas as pd
import scipy.optimize as opt
from itertools import accumulate

class ChoiceModels(object):
    
    '''
    This class defines methods that will be used later in speficying and estimating choice models.
    '''
   
    def load_data(self, path, file):

        if file.endswith('parquet'):
            df = pd.read_parquet(os.path.join(path, file))
        else:
            df = pd.read_csv(os.path.join(path, file))
        df = df.sort_values(by=['ID_household', 'ID_product'])
        
        df = self.create_choice_attributes(df)
        
        return df
    
    def create_choice_attributes(self, df):
        '''
        Construct dependent variable and ID_product attributes
        '''
        # Generate dependent variable
        df_copy = df.copy()
        df_copy['y'] = 0.
        df_copy.loc[(df_copy['ID_choice'] == df_copy['ID_product']), 'y'] = 1
        
        # Generate model year and current model year by household
        df_copy['ModelYear'] = df_copy['ID_product'].apply(lambda i: int(i.split(', ')[0]))
        df_copy['CurrentModelYear'] = df_copy.groupby('ID_household')['ModelYear'].transform('max')
        df_copy['vintage'] = df_copy['CurrentModelYear'] - df_copy['ModelYear']
        df_copy['newcar']=(df_copy['vintage'] <= 1).astype(int)
        #df_check = df_copy.query("ID_household<135").sort_values(['ID_household', 'ID_product'])[['ID_household', 'ID_product', 'ID_choice', 'ModelYear', 'CurrentModelYear', 'vintage', 'newcar']]
        
        # Generate vehicle type dummy
        df_copy = df_copy.query("Class != 'VAN'")
        df_copy = df_copy.join(pd.get_dummies(df_copy['Class'], prefix='class'))
        
        # Generate drive type dummy
        df_copy['Drivetype2'] = df_copy['Drive type'].map({'FWD':'FWD/RWD', 'RWD':'FWD/RWD', '4WD':'4WD/AWD', 'AWD':'4WD/AWD'})
        df_copy = df_copy.join(pd.get_dummies(df_copy['Drivetype2'], prefix='drive'))
        
        # Generate transmission dummy
        df_copy = df_copy.join(pd.get_dummies(df_copy['Transmission_Type'], prefix='trans'))
        
        # Generate warranty dummy
        df_copy['Warranty2'] = df_copy['Warranty_Basic'].map({'2 yr./ 24000 mi.':'less_36k', 
                                                              '3 yr./ 36000 mi.':'less_36k',
                                                              '3 yr./ 50000 mi.':'36k_60k',
                                                              '4 yr./ 50000 mi.':'36k_60k',
                                                              '4 yr./ 60000 mi.':'36k_60k',
                                                              '5 yr./ 60000 mi.':'36k_60k',
                                                              '5 yr./ 100000 mi.':'60k_more'})
        df_copy = df_copy.join(pd.get_dummies(df_copy['Warranty2'], prefix='warranty'))
        
        # Generate fuel type dummy
        #df_copy = df_copy.query("Fuel_Type != 'ELECTRIC'")
        df_copy = df_copy.join(pd.get_dummies(df_copy['Fuel_Type'], prefix='fuel'))
        
        # Generate size dummy
        df_copy['size_mid'] = df_copy['Class_Size'].str.contains('MIDSIZE', case=False)
        df_copy['size_large'] = df_copy['Class_Size'].str.contains('LARGE', case=False)
        df_copy[['size_mid', 'size_large']] = df_copy[['size_mid', 'size_large']].apply(lambda col: col.astype(int))
                               
        # Generate 4-door dummy
        df_copy['door_atleast_4'] = (df['Doors'] >= 4).astype(int)
        
        # Generate country dummy
        df_copy = df_copy.join(pd.get_dummies(df_copy['Country of origin'], prefix='country'))
        df_copy['country_others'] = np.where((df_copy['country_Germany'] + df_copy['country_Japan'] + df_copy['country_United States'] == 0), 1, 0)
        
        # Generate college or above dummy
        df_copy = df_copy.join(pd.get_dummies(df_copy['Education'], prefix='educ'))
        df_copy['educ_atleast_College'] = np.where(df_copy['Education']!='High_School', 1,0)
        
        # Generate income dummy
        df_copy['income_cat'] = np.select(
            [(df_copy['Income']=='Less than $15,000') | (df_copy['Income']=='$15,000 - $19,999') | (df_copy['Income']=='$20,000 - $29,999'),
             (df_copy['Income']=='$30,000 - $39,999') | (df_copy['Income']=='$40,000 - $49,999') | (df_copy['Income']=='$50,000 - $59,999'),
             (df_copy['Income']=='$60,000 - $69,999') | (df_copy['Income']=='$70,000 - $79,999') | (df_copy['Income']=='$80,000 - $89,999') | (df_copy['Income']=='$90,000 - $99,999'),
             (df_copy['Income']=='$100,000 - $124,999') | (df_copy['Income']=='$125,000 - $149,999') | (df_copy['Income']=='Greater than $149,999')],
            ['less_30', '30_60', '60_100', 'more_100'])
        df_copy = df_copy.join(pd.get_dummies(df_copy['income_cat'], prefix='inc'))
        
        # Generate average income 
        df_copy['income_avg'] = df_copy['Income'].replace('Less than $15,000', '$12,500 - $15,000')
        df_copy['income_avg'] = df_copy['income_avg'].replace('Greater than $149,999', '$149,999 - $175,000')
        df_copy['income_avg'] = df_copy['income_avg'].apply(lambda i: i.split(' - ')).apply(lambda l: [int(n.split('$')[1].replace(',','')) for n in l]).apply(lambda l: np.mean(l))
        
        # Generate price/income
        df_copy['log_MSRP/inc'] = np.log(df_copy['Base MSRP'] / df_copy['income_avg'])
        
        # Generate vehicle make dummy
        df_copy['Make'] = df_copy['ID_product'].apply(lambda i: i.split(', ')[1])
        df_copy = df_copy.join(pd.get_dummies(df_copy['Make'], prefix='make'))
        
        # Generate operating costs
        df_copy['MPG_Mixed'] = np.where(df_copy['Fuel_Type']=='ELECTRIC',
                                        df_copy['MPG_Mixed'] * 0.029669,
                                        df_copy['MPG_Mixed'])
                                            # Converts MPGe to miles/kWh for EVs        
        df_copy['cost_per_mile'] = df_copy['Price_G'] / df_copy['MPG_Mixed']
        df_copy['cost_per_mile'] = np.where(df_copy['Fuel_Type']=='ELECTRIC',
                                            df_copy['Price_E'] / df_copy['MPG_Mixed'],
                                            df_copy['cost_per_mile'])
                                                # Use electricity price for EVs        
        df_copy['log_cost_per_mile/inc'] = np.log(df_copy['cost_per_mile'] / df_copy['income_avg'])

        # Generate number of children
        df_copy['has_children'] = np.where((df_copy['Household_Size'] - df_copy['Number_Adults'] > 0), 1,0)

        return df_copy
    
    def rescale_variables(self, df, vdic=None):
        '''
        This function rescales variables in the data.
        '''
        if vdic is None:
            return df 
        
        for k, v in vdic.items():
            df[k] = df[k] * v
        return df
        
    def create_interactions(self, df, interact_list):
        '''
        Parameters
        ----------
        df : pandas data frame
            
        interact_list : a List
            The list contains pairs of variable names as tuples
        Returns
        -------
        df : pandas data frame after adding interactions
            
        xz_list : A list of created interactions
        '''
        xz_list = []
        if interact_list is None:
            return df, xz_list
        for item in interact_list:
            vname = item[0] + "_" + item[1]
            df[vname] = df[item[0]] * df[item[1]]
            xz_list.append(vname)
        return df, xz_list         
        
    def group_list(self, ulst, lst_size):
        return [ulst[start:end] for start, end in zip([0] + list(accumulate(lst_size))[:-1],
                                                          accumulate(lst_size))]
        
    def optimization(self, objfun, para):
        '''
        Parameters
        ----------
        objfun : a user defined objective function of para
            
        para : a 1-D array with the shape (k,), where k is the number of parameters.
        Returns
        -------
        dict
            A dictionary containing estimation results
        '''
        v = opt.minimize(objfun, x0=para, jac=True, method='BFGS',tol=1e-6, 
                          options={'maxiter': 1000, 'disp': True})  
        return {'log_likelihood':-1*v.fun, "Coefficients": v.x, "Var_Cov": v.hess_inv,'Jac': v.jac}

    def halton_sequence(self, ndraws, seed, randomize=True, shuffle=True, cut=100):
        '''
        This method generates Haton sequence for random drawing. 
        ndraws: an integer defining the length of generated sequece
        seed: a prime number
        
        Return: a 1-D array with the shape (ndraws,) 
        '''
        discard = random.randint(0, cut)
        n = ndraws + discard;     
        k = np.fix(np.log(n+1) / np.log(seed))
        phi = np.zeros(1)
        i = 1
        while i <= k+1:
            x = phi
            j = 1
            while j<seed:
                y = phi + (j/seed**i)
                x = np.concatenate((x, y))
                j += 1
            phi = x
            i += 1
            
        x=phi
        j=1
        while j<seed and len(x) < n:
            y=phi+(j/seed**i)
            x = np.concatenate((x, y))
            j += 1
        phi=x[discard:n]
        
        if randomize is True:
            phi = phi + np.random.rand()
            phi[phi>=1] -= 1
        
        if shuffle is True:
            random.shuffle(phi)
        return phi
    
class MultinomialLogit(ChoiceModels):
    
    def __init__(self, path, file, config):
        self._initialize_data(path, file, config)
    
    def _initialize_data(self, p, f, config): 
        df = super().load_data(p, f)
        df = super().rescale_variables(df, config['v_rescale'])
        df, xz = super().create_interactions(df, config['v_interactions'])
        self.X_list = config['attributes'] + xz
        
        y_grouped = df.groupby('ID_household')['y'].apply(list)
        y_grouped = y_grouped.to_dict()
        self.y_grouped = list(y_grouped.values())
        self.group_size = [len(i) for i in self.y_grouped]
        self.Xmat = df[self.X_list].to_numpy()
        self.X_grouped = super().group_list(self.Xmat, self.group_size)
        self.nobs = len(self.Xmat)
    
    @staticmethod
    def mnl_groupby(groupby_tuple):
        p = groupby_tuple[1] / np.sum(groupby_tuple[1])
        x = groupby_tuple[0] - p
        x = np.multiply(groupby_tuple[2], x[:, None])
        x = x.sum(axis=0)
        return (np.sum(groupby_tuple[0] * np.log(p)), x)
    
    def compute_common(self, para):
        '''
        This method defines the data log-likelihood from a Multinomial Logit.
        '''
        xb = np.dot(self.Xmat, para)
        xb = np.exp(xb)
        xb_grouped = super().group_list(xb, self.group_size)
        return list(map(self.mnl_groupby, zip(self.y_grouped, xb_grouped, self.X_grouped)))
        
    def mnl_log_likelihood(self, para):
        
        out = self.compute_common(para) 
        ll = (-1/self.nobs) * sum([i[0] for i in out])
        dl = (-1/self.nobs) * np.sum([i[1] for i in out], axis=0)
        return ll, dl
    
    def estimation(self, para):
        '''
        Parameters
        ----------
        para : array
            a 1-D array with the shape(k,), where k is the number of model parameters.
        Returns
        -------
        A dictionary of estimation results
        '''
        y = super().optimization(self.mnl_log_likelihood, para)
        y["Var_Cov"] = (1/len(self.Xmat)) * y["Var_Cov"]
        return y
    
if __name__ == '__main__':
    import time
    in_p =r"c:\\users\\jiay\\texas"
    data = "df_YMM_safety_10000_30.parquet"
    mnl = MultinomialLogit(in_p, data)
    bini = np.zeros(len(mnl.X_list))
    start = time.time()
    x = mnl.estimation(bini)
    end = time.time()
    print("time:", end - start)
    
    #x = np.load(os.path.join(out_p, results), allow_pickle=True)
    #x = x[()]
