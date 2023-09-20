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


class ScrapOptimization:
    def __init__(self, general_info,  sim_settings):
        self.optimierung_id = sim_settings.id
        self.sim_settings = sim_settings
        self.general_info = general_info
        self.experiment_info = {
            "target": general_info.target,
            "features": general_info.features
        }
        
    def optimize(self, total_quantity:float, chemi_component:list, selected_stahl_name:str, ns,supplier_quantity_hist,sim_id_hist):
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
        df_schrott = ns.df_schrott # pd.read_csv(self.general_info.scrap_dataset)
        df_price = df_schrott["price"].to_numpy().astype(np.float32)
        company_count = int(df_schrott[["company"]].nunique())
        
        # load the chemical dataframe
        df_chemi = pd.read_csv(self.general_info.chemi_dataset)
        df_chemi[self.general_info.chemi_names] = df_chemi[self.general_info.chemi_names].astype(np.float32)
        
        ############################# Optimization #############################
        constant_features_names, schrotte_features_names, kreislauf_schrotte_names, legierung_schrotte_names,fremd_schrotte_names = self.df_columns_name(df)
        
        length_fremdschrott = len(fremd_schrotte_names) 
        
        total_variable_length = length_fremdschrott * company_count  # the total number of variable parameters to optimize
        
        price_list = df_price[:total_variable_length]  # the price list of all the schrott
        
        ############################## Optimization Settings ##############################
        x_lower = np.zeros(total_variable_length)  # the lower bound of the variable parameters
        x_upper = np.ones(total_variable_length) * total_quantity # the upper bound of the variable parameters
        
        fremdschrotte_chemi_table = self.fremdschrott_chemi_table(df_chemi,fremd_schrotte_names,company_count)
        
        # left hand side of equality constraints
        aeq = np.array(fremdschrotte_chemi_table)
        
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

        # function to calculate the transport cost of xgboost version
        def sum_t3_xgb(x):
            summe = 0
            # quantity = np.array([sum(g) for g in list(grouper(x,company_count))])
            quantity = np.array([sum(x[i::company_count]) for i in range(len(x) // company_count)])
            for q in quantity:
                if q <= 50.0:
                    summe += 0.0
                elif q > 50.0 and q <= 100.0:
                    summe += 2.5 * (q-50) 
                else:
                    summe += 100.0
                return summe

        # objective xgboost: the total cost of the optimization problem
        def objective(x, constant_column, kreislauf_column, legierung_column):
            t1 = np.dot(x, price_list)
            list_fremdschrotte = [sum(g) for g in list(grouper(x,company_count))]
            features = np.concatenate((constant_column, kreislauf_column, list_fremdschrotte, legierung_column))
            t2 = f_xgb(features)*0.4
            t3 = sum_t3_xgb(x)
            #return (t1 + t2 + t3).item()
            return (t1+t3).item(), t2

        # ann prediction strompreis für indsutrie  40,11 ct/kWh
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
            #quantity = tf.reduce_sum(tf.reshape(x,[-1,company_count]), axis=1) 
            quantity = [tf.reduce_sum(tf.gather(x, range(i, len(x), company_count))) for i in range(len(x) // company_count)]
            for q in quantity:
                q = tf.reshape(q, ())
                summe += tf.cond(q <= 50.0, 
                                lambda: 0.0, 
                                lambda: tf.cond(q > 50.0 and q <= 100.0, 
                                                lambda: 2.5 * (q - 50), 
                                                lambda: 100.0))
            return summe

        # function for calculate the total cost, tf version
        @tf.function
        def objective_tf(x,constant_column,kreislauf_column,legierung_column):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            price_list_tf = tf.convert_to_tensor(price_list, dtype=tf.float32)
            t1 = tf.tensordot(x, price_list_tf,axes=1)
            t2 = tf_ann(x,constant_column,kreislauf_column,legierung_column)*0.4
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

        
        def optimize_grad(constant_column, kreislauf_column, legierung_column, beq, x_start,constraints, bounds, without=False):
            # Wrap the objective and gradient functions with lambda functions
            
            #wrapped_objective_tf = lambda x: objective_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
            #wrapped_grad_f_ann_tf = lambda x: grad_f_ann_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
            
            if without:
                wrapped_objective_tf = lambda x: objective_tf_without_strom(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
                wrapped_grad_f_ann_tf = None

            else:
                wrapped_objective_tf = lambda x: objective_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)  
                wrapped_grad_f_ann_tf = lambda x: grad_f_ann_tf(x.astype(np.float32), constant_column, kreislauf_column, legierung_column)
  


            # Initialize the dictionary to store the objective function values
            objective_values = {}

            # Callback function to store the objective function values at each iteration
            def callback(x, n_iter=[1]):
                objective_values[n_iter[0]] = wrapped_objective_tf(x).numpy().item()
                n_iter[0] += 1
            
            start = time.time()
            result = minimize(fun=wrapped_objective_tf, x0=x_start, jac=wrapped_grad_f_ann_tf, constraints=constraints, method='SLSQP',
                            options={'disp': False, 'maxiter':max_iter, 'ftol':1e-9}, bounds=bounds, tol=1e-9,
                            callback=callback)

            
            end = time.time()
            c_violation = (np.dot(aeq, result.x) - beq).tolist()
            
            elapsed_time = end - start
            return np.rint(result.x), result.fun, c_violation, elapsed_time, objective_values
        
        ############################## Opt Running ##############################
        opt_result = []
        
        # create a function to return the equality constraint for every index that lb >= ub
        def create_equality_fun_zero_for_index(idx):
            def equality_fun_zero(x):
                return x[idx]
            return equality_fun_zero
        
        
        for i in range(self.sim_settings.epochs):

            constant_column, kreislauf_column, legierung_column, chemi_to_achieve_fremdschrotte = self.calculate_chemi_component(df, df_chemi, constant_features_names,kreislauf_schrotte_names,legierung_schrotte_names,fremd_schrotte_names,total_chemi_to_achieve)
            
            # right hand side of equality constraint
            beq = chemi_to_achieve_fremdschrotte
            
            x_start = np.linalg.lstsq(aeq, beq, rcond=None)[0]
            print("################# Optimizing for SLSQP iteration #################")
                
            bounds = Bounds([0.0]*total_variable_length, ns.df_schrott["quantity"].to_list())
            def equality_fun(x):
                 return np.dot(aeq, x) - beq 
            constraints = [{'type': 'eq','fun' : equality_fun}]
            
            # update the equality constraints
            if np.any(bounds.lb >= bounds.ub):
                # get the index of the upper bound which is 0
                bounds_index = np.where(bounds.lb > bounds.ub)[0]
                print(f"bounds_index: {bounds_index}")
                # for every index, we add a new equality constraint
                for index in bounds_index:
                    constraints.append({'type': 'eq', 'fun': create_equality_fun_zero_for_index(index)})
                    bounds.lb[bounds_index] = -1.
                    bounds.ub[bounds_index] = 1.
                    
            
            x_ann, _, c_violation_ann, elapsed_time_ann, _  = optimize_grad(constant_column,kreislauf_column, legierung_column,beq, x_start, constraints, bounds,  without=without)
            print("################### original fremd schrotte ###################")
            print(ns.df_schrott["quantity"].to_list())
            # substract the optimal schrott list from the total quantity
            print("############### ANN result #################", x_ann)
            # fremd_schrotte.loc[:, "quantity"] = fremd_schrotte.loc[:,"quantity"].sub(x_ann)
            fremd_schrotte = ns.df_schrott.copy()
            subs = fremd_schrotte.loc[:,"quantity"].sub(x_ann)
            fremd_schrotte["quantity"] = subs
            ns.df_schrott = fremd_schrotte
            violence = np.sum(np.abs(c_violation_ann)) / np.sum(beq)
            
            
            if violence > self.general_info.violation_threshold:
                # return the message to the frontend
                _data = f"Simulation:{self.sim_settings.id}- violation is more than threshold:  {violence}{self.general_info.  violation_threshold}. Please try again."
                # terminate the optimization process
                print(_data)
                return
            else:
                # update and save the database of the schrott quantity
                try:
                    supplier_quantity_hist.append(ns.df_schrott["quantity"].to_list())
                    sim_id_hist.append(self.sim_settings.id)
                    result_current = {}
                    optimal_value, schmelz_preis = objective(x_ann, constant_column, kreislauf_column,legierung_column)
                    result_current["total_cost"] = optimal_value + schmelz_preis
                    result_current["optimal_cost"] = optimal_value
                    result_current["schmelz_preis"] = schmelz_preis
                    result_current['optimal_schrott_list'] = x_ann.tolist()
                    result_current['elapsed_time'] = elapsed_time_ann
                        
                    # if opt_result:
                    #     if sum(objective_values.values()) < sum(opt_result[0]["objective_values"].value()):
                    #         opt_result[0] = result_current
                    # else:
                    #     opt_result.append(result_current)
                    
                    opt_result.append(result_current)
                    
                except Exception as e:
                    print(f"Simulation:{self.sim_settings.id}- Exception Happened: The database is notupdated. Please try again.")
                    print(traceback.format_exc())
                finally:
                    _data = {
                        "optimierung_id": self.optimierung_id,
                        "opt_result": opt_result,
                    }
                    with open(f'buy_test/buy_list_{self.sim_settings.id}.json', 'w') as f:
                        json.dump(_data, f, indent=4)


    def fremdschrott_chemi_table(self, df_chemi, fremd_schrotte_names,company_count):
        # construct the chemical table
        # assume that every company's chemical elements for every schrott is identical
        df_chemi_fremdschrott= (df_chemi[self.general_info.chemi_names].iloc[:len(fremd_schrotte_names)])
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
        #print(f"{constant_column=}")
        
        # calculate the chemical component for kreislauf
        kreislauf_column = (df_random_row[kreislauf_schrotte_names].values[0]).astype(np.float32)
        kreislauf_chemical_table = df_chemi[self.general_info.chemi_names].iloc[len(kreislauf_schrotte_names)-1:]
        chemi_component_kreislauf = (np.dot(kreislauf_column, kreislauf_chemical_table) /100.0).astype(np.float32)
        
        # calculate the chemical component for legierungen
        legierung_column = (df_random_row[legierung_schrotte_names].values[0]).astype(np.float32)
        legierung_chemical_table = df_chemi[self.general_info.chemi_names].iloc[len(fremd_schrotte_names):len(kreislauf_schrotte_names)-1]
        chemi_component_legierung = (np.dot(legierung_column, legierung_chemical_table) /100.0).astype(np.float32)

        
        # calculate the chemical compoent for fremdschrotte
        chemi_to_achieve_fremdschrotte = (np.abs(total_chemi_to_achieve - chemi_component_kreislauf - chemi_component_legierung)).astype(np.float32)
        
        return constant_column, kreislauf_column, legierung_column, chemi_to_achieve_fremdschrotte
