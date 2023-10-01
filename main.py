from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import traceback
import os
import pickle
import time
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from pydantic import BaseSettings, BaseModel, validator
from joblib import Parallel, delayed


from scrap_optimization_test import ScrapOptimization


def my_fact(x):
    print(x)
    return x

class GeneralInfo(BaseModel):
    features:list
    target: str
    chemi_names: list
    train_dataset: str
    scrap_dataset: str
    chemi_dataset:str
    ML_MODELS:str
    violation_threshold:float
    electricity_price:float
    transport_coefficents:list[float]
    comeback_epoches:int # after how many epoches add comeback_value
    comeback_value:int


    @validator("electricity_price", pre=True)
    def random_select_electricity_prices(cls, v):
        return random.choice(v)
    
    @validator("transport_coefficents", pre=True)
    def random_select_transport_coefficents(cls, v):
        return random.choice(v)


class SimulationInfo(BaseModel):
    id: int
    steel_type: str
    total_quantity:int = np.random.randint(5000, 7000)
    epochs: int = np.random.randint(4, 15)
    feature: int

class Settings(BaseSettings):
    general_info:GeneralInfo
    simulations: list[SimulationInfo]


settings = Settings.parse_file("settings.json")
simulations = settings.simulations
general_info = settings.general_info
supplier_quantity_hist = []
df_schrott = pd.read_csv(general_info.scrap_dataset)
df_schrott = df_schrott[df_schrott["name"].str.startswith("F")]
global_lock = multiprocessing.Lock()

report_dir_name = time.strftime("%Y%m%d-%H%M%S")
report_dir = f"sim_output/{report_dir_name}"

general_info.electricity_price=0.6
general_info.transport_coefficents = [30.0,60.0,70.0]


def run_simulation(simulation):
    simulation_settings,df_schrott,supplier_quantity_hist, sim_id_hist = simulation

    try: 
        # global_lock.acquire()
        so= ScrapOptimization(general_info, simulation_settings)
        steel_chemi_df = pd.read_csv("assets/steel_chemi_components.csv")
        chemies = steel_chemi_df.loc[steel_chemi_df["name"] == float(simulation_settings.steel_type)]
        chemies = chemies[['C','Si','Mn','Cr','Mo','V']].values
        chemi_component = [float(i) for i in chemies[0]] 
        print(f"Running optimization for id:{simulation_settings.id}")
                
        simulation_settings.total_quantity = np.random.randint(5000, 7000)

        so.optimize(simulation_settings.total_quantity, chemi_component, simulation_settings.steel_type, df_schrott, supplier_quantity_hist,sim_id_hist)
        # global_lock.release()
    except Exception as e:
        print("Error:")
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{report_dir}/settings.json", "w") as f:
        data ={
            "scrap_dataset":general_info.scrap_dataset,
            "violation_threshold":general_info.violation_threshold,
            "electricity_price": general_info.electricity_price,
            "transport_coefficient": general_info.transport_coefficents,
            "comeback_epoche": general_info.comeback_epoches,
            "comeback_value": general_info.comeback_value,
        }
        json.dump(data,f,indent=4)
    initial_df_schrott = df_schrott.copy()
    with multiprocessing.Manager() as manager:
        # lock = manager.Lock()
        supplier_quantity_hist = manager.list()
        sim_id_hist = manager.list()

        ns = manager.Namespace()
        ns.df_schrott = df_schrott
        ns.comeback_value = settings.general_info.comeback_value
        nonzero_quantities = ns.df_schrott["quantity"].to_numpy().nonzero()[0].tolist()
        print("Nonzero indices: ", nonzero_quantities)
        comeback_info = {index: (settings.general_info.comeback_epoches, True) for index in nonzero_quantities}
        ns.comeback_info = comeback_info

        simulations = [(item, ns, supplier_quantity_hist, sim_id_hist) for item in simulations]
        supplier_quantity_hist.append(ns.df_schrott["quantity"].to_list())
        core_count = 8
        with Pool(core_count) as p:
            p.map(run_simulation, simulations)

        print("len(supplier_quantity_hist): ",len(supplier_quantity_hist))
        all_hist = np.empty(shape=(0,len(supplier_quantity_hist)))
        fig, axes = plt.subplots(1,2, figsize=(15, 7))
        for idx, remote_scrap in enumerate(supplier_quantity_hist[0]):
            values = []
            for i in range(len(supplier_quantity_hist)):
                values.append(supplier_quantity_hist[i][idx])

            all_hist = np.vstack((all_hist, np.array(values).reshape(1,-1).copy()))


        for idx, row in enumerate(all_hist):
            if not np.all(np.isclose(row, row[0], atol=0.25)) and not np.all(row[:10]<1) and np.nan not in row:
                axes[0].plot(range(row.shape[0]), row, label=f"F{idx+1}",linewidth=0.5)
        np.save("npy/saved_plt.npy", all_hist)

        print(f"sim_id_hist= {[item for item in sim_id_hist]}")
        counts = [0]*8
        for item in sim_id_hist:
            counts[item-1] += 1

        axes[1].bar(range(core_count), counts)
        
        plt.savefig(f"{report_dir}/simulation_output6.png")
        plt.show()

        counter = [0] * 10
        for idx, item in enumerate(initial_df_schrott["quantity"]):
            if item != ns.df_schrott["quantity"][idx]:
                counter_idx = idx % 10
                counter[counter_idx] += 1

        print(f"{np.count_nonzero(np.array(counter))} supplier used!")


        