import numpy as np
import pandas as pd
from pydantic import BaseSettings

from scrap_optimization import ScrapOptimization



class Settings(BaseSettings):
    steel_type: str
    total_quantity:int
    features:list
    target: str
    train_dataset: str
    scrap_dataset: str
    chemi_dataset:str
    ML_MODELS:str



settings = Settings.parse_file("settings.json")


if __name__ == "__main__":
    so = ScrapOptimization(1, settings)

    steel_chemi_df = pd.read_excel("assets/steel_chemi_components.xlsx")
    chemies = steel_chemi_df.loc[steel_chemi_df["name"] == float(settings.steel_type)]
    chemies = chemies[['C','Si','Mn','Cr','Mo','V']].values
    chemi_component = [float(i) for i in chemies[0]] 

    so.optimize(settings.total_quantity, chemi_component, settings.steel_type)
