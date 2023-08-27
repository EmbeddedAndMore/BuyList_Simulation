from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import traceback
import os
import pickle


import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from pydantic import BaseSettings, BaseModel
from joblib import Parallel, delayed

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from scrap_optimization import ScrapOptimization


class GeneralInfo(BaseModel):
    features:list
    target: str
    chemi_names: list
    train_dataset: str
    scrap_dataset: str
    chemi_dataset:str
    ML_MODELS:str
    violation_threshold:float

class SimulationInfo(BaseModel):
    id: int
    steel_type: str
    total_quantity:int
    epochs: int
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

# physical_devices = tf.config.list_physical_devices('CPU') 
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)


def run_simulation(simulation):
    simulation_settings,df_schrott,supplier_quantity_hist, sim_id_hist = simulation

    try:
        so = ScrapOptimization(general_info, simulation_settings)
        steel_chemi_df = pd.read_csv("assets/steel_chemi_components.csv")
        chemies = steel_chemi_df.loc[steel_chemi_df["name"] == float(simulation_settings.steel_type)]
        chemies = chemies[['C','Si','Mn','Cr','Mo','V']].values
        chemi_component = [float(i) for i in chemies[0]] 
        print(f"Running optimization for id:{simulation_settings.id}")
        so.optimize(simulation_settings.total_quantity, chemi_component, simulation_settings.steel_type, df_schrott, supplier_quantity_hist,sim_id_hist)
    except Exception as e:
        print("Error:")
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    # # print(settings)
    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        supplier_quantity_hist = manager.list()
        sim_id_hist = manager.list()
        ns = manager.Namespace()
        ns.df_schrott = df_schrott

        simulations = [(item, ns, supplier_quantity_hist, sim_id_hist) for item in simulations]
        supplier_quantity_hist.append(ns.df_schrott["quantity"].to_list())
        core_count = 8
        with Pool(core_count) as p:
            p.map(run_simulation, simulations)

        print("len(supplier_quantity_hist): ",len(supplier_quantity_hist))
        all_hist = np.empty(shape=(0,len(supplier_quantity_hist)))
        # values_list = []
        fig, axes = plt.subplots(1,2, figsize=(15, 7))
        for idx, remote_scrap in enumerate(supplier_quantity_hist[0]):
            values = []
            for i in range(len(supplier_quantity_hist)):
                values.append(supplier_quantity_hist[i][idx])

            all_hist = np.vstack((all_hist, np.array(values).reshape(1,-1).copy()))


        for idx, row in enumerate(all_hist):
            if not np.all(np.isclose(row, row[0], atol=0.25)) and not np.all(row[:10]<1) and np.nan not in row:
                axes[0].plot(range(row.shape[0]), row, label=f"F{idx+1}",linewidth=0.5)
        np.save("saved_plt.npy", all_hist)

        print(f"sim_id_hist= {[item for item in sim_id_hist]}")
        counts = [0]*8
        for item in sim_id_hist:
            counts[item-1] += 1

        axes[1].bar(range(core_count), counts)
        plt.savefig(f"sim_output/simulation_output2.png")
        plt.show()


        