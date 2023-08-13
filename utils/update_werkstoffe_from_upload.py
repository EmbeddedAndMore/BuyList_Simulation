import pandas as pd
from sqlalchemy import create_engine
from django.conf import settings


def upload_werkstoffe(foreign_key: int, file_path: str, table_name):
    """
    uploads the werkstoffe data from excel file and saves it to the database

    :param foreign_key: foreign key of the table
    :param file_path: path to the excel file
    :param table_name: name of the table   e.g. Tiefzieh_stanzabfaelle._meta.db_table
    """
    df = pd.read_excel(file_path)

    df.insert(0, 'spezifikation_id', foreign_key)

    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    database_name = settings.DATABASES['default']['NAME']

    database_url = f"postgresql://{user}:{password}@localhost:5432/{database_name}"
    engine = create_engine(database_url)

    df.to_sql(table_name, engine, if_exists='append', index=False)
    
    

