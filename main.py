from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pydantic import BaseSettings, BaseModel


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


def run_simulation(simulation_settings):
    try:
        so = ScrapOptimization(general_info, simulation_settings)
        steel_chemi_df = pd.read_csv("assets/steel_chemi_components.csv")
        chemies = steel_chemi_df.loc[steel_chemi_df["name"] == float(simulation_settings.steel_type)]
        chemies = chemies[['C','Si','Mn','Cr','Mo','V']].values
        chemi_component = [float(i) for i in chemies[0]] 
        print(f"Running optimization for id:{simulation_settings.id}")
        so.optimize(simulation_settings.total_quantity, chemi_component, simulation_settings.steel_type)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # # print(settings)
    with ProcessPoolExecutor(max_workers=8) as executor:
         executor.map(run_simulation, simulations)
    # run_simulation(simulations[0])


        