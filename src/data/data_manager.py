from typing import Dict

import pandas as pd
from loguru import logger
from pandas import DataFrame

class DataManager:
    def __init__(self, service, config):
        self.service = service
        self.config = config
        self.data: Dict[str, DataFrame] = {}
        self.default_dir = "src/data"
        self.validation_data: DataFrame = DataFrame()

    def load_data(self):
        logger.info("Starting to load data")
        for language in self.config["DATA"]['LANGUAGES']:
            df = pd.read_csv(f"{self.default_dir}/{language}-Alerts.csv")
            self.data[language] = df

        # Add validation data
        logger.info("Adding validation data")
        validation_df = pd.read_csv(f"{self.default_dir}/{self.config['DATA']['VALIDATION']}.csv")
        self.validation_data = validation_df

        # Check if data is loaded
        if len(self.data) == 0:
            logger.error("No data loaded")
            raise Exception("No data loaded")
        else:
            logger.success("Data loaded successfully")





