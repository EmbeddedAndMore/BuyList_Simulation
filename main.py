from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pydantic import BaseSettings, BaseModel


from scrap_optimization import ScrapOptimization



class SimulationInfo(BaseModel):
    id: int
    steel_type: str
    total_quantity:int
    features:list
    target: str
    train_dataset: str
    scrap_dataset: str
    chemi_dataset:str
    ML_MODELS:str
    epochs: int

class Settings(BaseSettings):
    simulations: list[SimulationInfo]



settings = Settings.parse_file("settings.json").simulations


def run_simulation(simulation_settings):
    try:
        so = ScrapOptimization(simulation_settings)
        steel_chemi_df = pd.read_excel("assets/steel_chemi_components.xlsx")
        chemies = steel_chemi_df.loc[steel_chemi_df["name"] == float(simulation_settings.steel_type)]
        chemies = chemies[['C','Si','Mn','Cr','Mo','V']].values
        chemi_component = [float(i) for i in chemies[0]] 
        print(f"Running optimization for id:{simulation_settings.id}")
        so.optimize(simulation_settings.total_quantity, chemi_component, simulation_settings.steel_type)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # print(settings)
    with ProcessPoolExecutor(max_workers=8) as executor:
         executor.map(run_simulation, settings)



        