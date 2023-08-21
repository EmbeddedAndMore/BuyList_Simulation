import os
import time
import json
import traceback
from datetime import datetime


import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import cobyqa
import matplotlib.pyplot as plt
from pdfo import pdfo
from scipy.optimize import Bounds
from scipy.optimize import minimize
from more_itertools import grouper


chemi_names = ['C','Si','Mn','Cr','Mo','V']


class ScrapOptimization:
    def __init__(self, general_info,  sim_settings):
        self.optimierung_id = sim_settings.id
        self.sim_settings = sim_settings
        self.general_info = general_info
        self.experiment_info = {
            "target": general_info.target,
            "features": general_info.features
        }
        
    def optimize(self, total_quantity:float, chemi_component:list, selected_stahl_name:str):
        """
        Args:
            total_quantity (int): the total amount of scraps to optimize in kg 
            chemi_component (list): the list of chemical components to optimize
            selected_stahl_name (str): the name of the final steel product to optimize (1.2343, 1.2344,1.2379,1.3343)
        """
        
        # chemi component and total weight of the final steel product to optimize
        chemi_component = (np.array(chemi_component) / 100.0).astype(np.float32)
        print(chemi_component)
        total_chemi_to_achieve = total_quantity * chemi_component
        
        # Load training data
        df = pd.read_csv(self.general_info.train_dataset)
        df = df[self.general_info.features]   # extract the used features from the dataframe
        
        # convert the `Schrott` model to a pandas dataframe, and extract the `price` column
        df_schrott = pd.read_csv(self.general_info.scrap_dataset)
        df_price = df_schrott["price"].to_numpy().astype(np.float32)
        company_count = int(df_schrott[["company"]].nunique())
        
        # load the chemical dataframe
        df_chemi = pd.read_excel(self.general_info.chemi_dataset)
        df_chemi[chemi_names] = df_chemi[chemi_names].astype(np.float32)
        
        ############################# Optimization #############################
        constant_features_names, schrotte_features_names, kreislauf_schrotte_names, legierung_schrotte_names,fremd_schrotte_names = self.df_columns_name(df)
        
        length_fremdschrott = len(fremd_schrotte_names) 
        total_variable_length = length_fremdschrott * company_count  # the total number of variable parameters to optimize
        
        price_list = df_price[:total_variable_length]  # the price list of all the schrott
        
        ############################## Optimization Settings ##############################
        x_lower = np.zeros(total_variable_length)  # the lower bound of the variable parameters
        x_upper = np.ones(total_variable_length) * total_quantity # the upper bound of the variable parameters
        
        fremdschrotte_chemi_table = self.fremdschrott_chemi_table(df_chemi,fremd_schrotte_names,company_count)
        
        # right hand side of equality constraints
        aeq = np.array(fremdschrotte_chemi_table)
        
        # bounds of scipy constraints 
        # TODO: the actual quantity of fremdschrott available quantity in the market, should be remove to the loop
        # bounds = Bounds([0.0]*total_variable_length, [max(total_chemi_to_achieve)]*total_variable_length)
        
        # max iteration 
        max_iter = 300
        ############################## Optimization Settings ##############################
        
        ############################## ML Settings ##############################
        xgb_model = xgb.Booster()
        # based on the selected steel name, load the corresponding ml model (1.2343, 1.2344,1.2379,1.3343)
        xgb_model.load_model(os.path.join(self.general_info.ML_MODELS, f'{selected_stahl_name}/XGB.json'))
        ann_model = tf.keras.models.load_model(os.path.join(self.general_info.ML_MODELS, f'{selected_stahl_name}/ANN'))
        
        # function for xgb prediction
        def f_xgb(x):
            y = xgb_model.predict(xgb.DMatrix([x]))   #x1.reshape(1,-1))
            return y.item()

        # function to calculate the total cost of xgboost version
        def sum_t3_xgb(x):
            """return sum of three objectives"""
            summe = 0
            quantity = np.array([sum(g) for g in list(grouper(x,company_count))])
            for q in quantity:
                if q <= 10.0:
                    summe += 0.0
                elif q > 10.0 and q <= 50.0:
                    summe += 2.5 * (q-10) 
                else:
                    summe += 100.0
                return summe

        # objective xgboost
        def objective(x, constant_column, kreislauf_column, legierung_column):
            t1 = np.dot(x, price_list)
            # print(f"{company_count=}")
            # print(f"x: {x.shape}")
            list_fremdschrotte = [sum(g) for g in list(grouper(x,company_count))]
            features = np.concatenate((constant_column, kreislauf_column, list_fremdschrotte, legierung_column))
            t2 = f_xgb(features)
            t3 = sum_t3_xgb(x)
            return (t1 + t2 + t3).item()

        # ann prediction 
        @tf.function
        def tf_ann(x,constant_column,kreislauf_column,legierung_column):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            constant_column_tf = tf.convert_to_tensor(constant_column, dtype=tf.float32)
            kreislauf_column_tf = tf.convert_to_tensor(kreislauf_column, dtype=tf.float32)
            legierung_column_tf = tf.convert_to_tensor(legierung_column, dtype=tf.float32)

            list_fremdschrotte = tf.reduce_sum(tf.reshape(x,[-1,company_count]), axis=1)#tf.convert_to_tensor([sum(g) for g in list(grouper(10, x))])
            x = tf.concat([constant_column_tf, kreislauf_column_tf, list_fremdschrotte, legierung_column_tf], axis=0)
            x = tf.convert_to_tensor([x])
            ann_pred = ann_model(x)[0]
            return ann_pred
        
        # function to calculate the logistic cost of tf version
        @tf.function
        def sum_t3_tf(x):
            """
            X: tf.Tensor, the quantity of every scrap
            """
            summe = tf.constant(0.0, dtype=tf.float32)
            quantity = tf.reduce_sum(tf.reshape(x,[-1,company_count]), axis=1) #tf.convert_to_tensor([sum(g) for g in list(grouper(10, x))])

            for q in quantity:
                q = tf.reshape(q, ())
                summe += tf.cond(q <= 10.0, 
                                lambda: 0.0, 
                                lambda: tf.cond(q > 10.0 and q <= 50.0, 
                                                lambda: 2.5 * (q - 10), 
                                                lambda: 100.0))
            return summe

        # function for calculate the total cost, tf version
        @tf.function
        def objective_tf(x,constant_column,kreislauf_column,legierung_column):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            price_list_tf = tf.convert_to_tensor(price_list, dtype=tf.float32)
            t1 = tf.tensordot(x, price_list_tf,axes=1)
            t2 = tf_ann(x,constant_column,kreislauf_column,legierung_column)
            t3 = sum_t3_tf(x)
            
            return t1 + t2 + t3

        # function to calculate the jacobian of tf 
        @tf.function
        def grad_f_ann_tf(x,constant_column,kreislauf_column,legierung_column):
            x1 = tf.convert_to_tensor(x, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x1)
                t2 = objective_tf(x1,constant_column,kreislauf_column,legierung_column)
                y = tape.jacobian(t2, x1)
                return y
            
        ############################## ML Settings ##############################
        
        ############################## Opt Running ##############################
        
        def optimize_cobyqa(constant_column, kreislauf_column, legierung_column, beq, x_start):
            # Wrap the objective function with a lambda to include the constant variables
            wrapped_objective = lambda x: objective(x, constant_column, kreislauf_column, legierung_column)
            
            start_cobyqa = time.time()
            res_cobyqa = cobyqa.minimize(wrapped_objective, x0=x_start, xl=x_lower, xu=x_upper,
                                        aeq=aeq, beq=beq, options={"disp": False, "maxiter": max_iter})
            end_cobyqa = time.time()

            elapsed_time_cobyqa = end_cobyqa - start_cobyqa
            
            c_violation = (np.dot(aeq, res_cobyqa.x) - beq).tolist()

            return res_cobyqa.x, res_cobyqa.fun, c_violation, elapsed_time_cobyqa
        
        def optimize_pdfo(constant_column, kreislauf_column, legierung_column, beq, x_start):
            wrapped_objective = lambda x: objective(x, constant_column, kreislauf_column, legierung_column)
            
            start_pdfo = time.time()

            def nlc_eq(x):
                return np.dot(aeq, x) - beq

            nonlin_con_eq = {'type': 'eq', 'fun': nlc_eq} 

            res_pdfo = pdfo(wrapped_objective, x_start, bounds=bounds, constraints=[nonlin_con_eq],
                        options={'maxfev': max_iter})

            end_pdfo = time.time()

            elapsed_time_pdfo = end_pdfo - start_pdfo
            
            #c_violation = (np.dot(aeq, res_cobyqa.x) - beq).tolist()
            c_violation = res_pdfo.constr_value[0].tolist()

            return res_pdfo.x, res_pdfo.fun, c_violation, elapsed_time_pdfo, res_pdfo.method
        
        # TODO: add the bounds
        def optimize_grad(constant_column, kreislauf_column, legierung_column, beq, x_start, bounds):
            # Wrap the objective and gradient functions with lambda functions
            
            wrapped_objective_tf = lambda x: objective_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
            wrapped_grad_f_ann_tf = lambda x: grad_f_ann_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
                
            def equality_fun(x):
                return np.dot(aeq, x) - beq 
            
            eq_cons = {'type': 'eq','fun' : equality_fun}
            
            
            # Initialize the dictionary to store the objective function values
            objective_values = {}

            # Callback function to store the objective function values at each iteration
            def callback(x, n_iter=[1]):
                objective_values[n_iter[0]] = wrapped_objective_tf(x).numpy().item()
                n_iter[0] += 1
            
            start = time.time()
            result = minimize(fun=wrapped_objective_tf, x0=x_start, jac=wrapped_grad_f_ann_tf, constraints=[eq_cons], method='SLSQP',
                            options={'disp': False, 'maxiter':max_iter, 'ftol':1e-9}, bounds=bounds, tol=1e-9,
                            callback=callback)

            
            end = time.time()
            c_violation = (np.dot(aeq, result.x) - beq).tolist()
            
            elapsed_time = end - start
            return result.x, result.fun, c_violation, elapsed_time, objective_values
        
        ############################## Opt Running ##############################
        opt_result = []
        all_results = []
        supplier_quantity_hist = []
        
        # check if the optimal schrott list is valid
        fremd_schrotte = df_schrott[df_schrott["name"].str.startswith("F")].copy()
        is_negative = False
        supplier_quantity_hist.append(fremd_schrotte["quantity"].to_list())
        
        
        for i in range(self.sim_settings.epochs):

            constant_column, kreislauf_column, legierung_column, chemi_to_achieve_fremdschrotte = self.calculate_chemi_component(df, df_chemi,
                                                                                                                        constant_features_names,kreislauf_schrotte_names,legierung_schrotte_names,fremd_schrotte_names,total_chemi_to_achieve)
        
            beq = chemi_to_achieve_fremdschrotte
            x_start = np.linalg.lstsq(aeq, beq, rcond=None)[0]
            print("################# Optimizing for SLSQP iteration #################")
            
            # TODO: add the bounds here
            bounds = ...
            x_ann, loss_ann, c_violation_ann, elapsed_time_ann, objective_values = optimize_grad(constant_column, kreislauf_column, legierung_column,beq, x_start, bounds)
            print("################### original fremd schrotte ###################")
            print(fremd_schrotte["quantity"].to_list())
            # substract the optimal schrott list from the total quantity
            # TODO:remove this condition
            x_ann = np.where(x_ann > 10, x_ann, 0)
            print("############### ANN result #################", x_ann)
            fremd_schrotte.loc[:, "quantity"] = fremd_schrotte.loc[:,"quantity"].sub(x_ann)
            
            # TODO: here should be the termination condition instead negative
            # if np.sum(np.abs(c_violation_ann)) / np.sum(beq) > 0.2:  first try 20%
            is_negative = any(fremd_schrotte["quantity"] < 0)
            print("fremd_schrotte", fremd_schrotte["quantity"].to_list())
            print("------- is negative", is_negative)
            if is_negative:
                # return the message to the frontend
                _data = f"Simulation:{self.sim_settings.id}- The scrap provider does not have enough scrap to provide. Please try again."
                # terminate the optimization process
                print(_data)
            else:
                # update and save the database of the schrott quantity
                try:
                    supplier_quantity_hist.append(fremd_schrotte["quantity"].to_list())
                    result_current = {}
                    
                    optimal_value = objective(x_ann, constant_column, kreislauf_column, legierung_column)
                    print("objective value", optimal_value)
                    
                    result_current['optimal_value'] = optimal_value
                    result_current['optimal_schrott_list'] = x_ann.tolist()
                    result_current['objective_values'] = objective_values
                    result_current['elapsed_time'] = elapsed_time_ann
                    if opt_result:
                        if sum(objective_values.values()) < sum(opt_result[0]["objective_values"].values()):
                            opt_result[0] = result_current
                            fremd_schrotte.to_csv(f"scrap_result_after_buy_{self.sim_settings.id}.csv")
                    else:
                        opt_result.append(result_current)
                        fremd_schrotte.to_csv(f"scrap_result_after_buy_{self.sim_settings.id}.csv")

                    
                    all_results.append(result_current)

                    
                    _data = {
                        "optimierung_id": self.optimierung_id,
                        "opt_result": opt_result,
                    }
                    
                except Exception as e:
                    print(f"Simulation:{self.sim_settings.id}- Exception Happened: The database is not updated. Please try again.")
                    print(traceback.format_exc())
                    
                finally:
                    with open(f'buy_list_{self.sim_settings.id}.json', 'w') as f:
                        json.dump(_data, f, indent=4)
        for idx, remote_scrap in enumerate(supplier_quantity_hist[0]):
            values = []
            for i in range(len(supplier_quantity_hist)):
                values.append(supplier_quantity_hist[i][idx])
            plt.plot(range(len(supplier_quantity_hist)), values, label=f"F{idx+1}")
            
        plt.legend()
        plt.savefig(f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        plt.show()

    def fremdschrott_chemi_table(self, df_chemi, fremd_schrotte_names,company_count):
        # construct the chemical table
        # assume that every company's chemical elements for every schrott is identical
        df_chemi_fremdschrott= (df_chemi[chemi_names].iloc[:len(fremd_schrotte_names)])
        fremdschrott_chemi = df_chemi_fremdschrott.T 
        n_times = company_count - 1
        temp_dfs = []
        
        for col_name in fremdschrott_chemi.columns:
            temp_df = fremdschrott_chemi[[col_name]].copy()
            for i in range(1, n_times+1):
                temp_df[f'{col_name}{i}'] = fremdschrott_chemi[col_name]
            temp_dfs.append(temp_df)

        fremdschrotte_chemi_table = pd.concat(temp_dfs, axis=1) / 100.0
        
        return fremdschrotte_chemi_table



    def df_columns_name(self, df):
        """
        return constant features, schrotte features, kreislauf schrotte, legierung schrotte
        """
        features_columns = df.columns.tolist()
        # remove_columns = ['HeatID', 'HeatOrderID','Energy']
        # features_columns = [x for x in columns if x not in remove_columns]
        
        # constant process parameter, start with "Feature"
        constant_features_names = [x for x in features_columns if "Feature" in x]
        
        #"K" means Kreislauf, "L" means Legierungen, "F" means Fremdschrotte, we only want to optimize "F"
        schrotte_features_names = [x for x in features_columns if "Feature" not in x]
        
        # schrotte name for Kreislauf
        kreislauf_schrotte_names = [x for x in features_columns if "K" in x]
        
        # schrotte name for legierung
        legierung_schrotte_names = [x for x in features_columns if "L" in x]
        
        # schrotte name for Fremdschrotte
        fremd_schrotte_names = [x for x in features_columns if "F" in x and len(x) < 4]
        
        return constant_features_names, schrotte_features_names, kreislauf_schrotte_names,legierung_schrotte_names,fremd_schrotte_names

    def calculate_chemi_component(self, df, df_chemi,
                                constant_features_names,
                                kreislauf_schrotte_names,
                                legierung_schrotte_names,
                                fremd_schrotte_names,
                                total_chemi_to_achieve):
        
        """
        return the randomly chosen row, and its constant column, kreislauf column and 
        legierung column, use them to return the chemical component of fremdschrotte column
        """
        df_random_row = df.sample()
        
        # calculate the constant features
        selected_feature = df.iloc[[self.sim_settings.feature]]
        constant_column = (selected_feature[constant_features_names].values[0]).astype(np.float32)
        print(f"{constant_column=}")
        
        # calculate the chemical component for kreislauf
        kreislauf_column = (df_random_row[kreislauf_schrotte_names].values[0]).astype(np.float32)
        kreislauf_chemical_table = df_chemi[chemi_names].iloc[len(kreislauf_schrotte_names)-1:]
        chemi_component_kreislauf = (np.dot(kreislauf_column, kreislauf_chemical_table) /100.0).astype(np.float32)
        
        # calculate the chemical component for legierungen
        legierung_column = (df_random_row[legierung_schrotte_names].values[0]).astype(np.float32)
        legierung_chemical_table = df_chemi[chemi_names].iloc[len(fremd_schrotte_names):len(kreislauf_schrotte_names)-1]
        chemi_component_legierung = (np.dot(legierung_column, legierung_chemical_table) /100.0).astype(np.float32)
        
        # calculate the chemical compoent for fremdschrotte
        chemi_to_achieve_fremdschrotte = (np.abs(total_chemi_to_achieve - chemi_component_kreislauf - chemi_component_legierung)).astype(np.float32)
        
        return constant_column, kreislauf_column, legierung_column, chemi_to_achieve_fremdschrotte
