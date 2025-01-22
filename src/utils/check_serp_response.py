import os
from dotenv import load_dotenv
import requests
import pandas as pd
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from typing import Optional

def check_api_keys(path_to_file="../../api_keys_status.xlsx"):
    load_dotenv()

    # Read the existing data
    df = pd.read_excel(path_to_file)

    for index, row in df.iterrows():
        key_name = row['Name']
        api_key = os.getenv(key_name)

        if api_key:
            try:
                response = requests.get(
                    f"https://serpapi.com/account?api_key={api_key}"
                )

                if response.status_code == 200:
                    data = response.json()
                    df.at[index, 'Status'] = data["total_searches_left"]
                else:
                    print(
                        f"Error checking {key_name}: status {response.status_code}"
                    )

                # if some problems with api
                # sleep(0.2)

            except Exception as e:
                print(f"Error checking {key_name}: {str(e)}")

    # Save the updated data back to the file
    df.to_excel(path_to_file, index=False)

    return df

class APIKeyManager:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.start_scheduler()
        self.path_to_file = "../../api_keys_status.xlsx"

    def start_scheduler(self):
        """Starts the scheduler with an update every hour"""
        self.scheduler.add_job(
            check_api_keys,
            'interval',
            hours=1,
            id='update_api_keys_status',
            next_run_time=datetime.now()  # Run the first time immediately
        )
        self.scheduler.start()

    def get_best_api_key(self) -> Optional[str]:
        """
        Reads the status file and returns the name of the key with the highest number of available requests,
        ignoring SERP_KEY_5
        """
        try:
            # Check if the file exists
            if not os.path.exists(self.path_to_file):
                print(f"File {self.path_to_file} does not exist")

            # Read the dataframe
            df = pd.read_excel(self.path_to_file)

            # Exclude SERP_KEY_5 and find the key with the maximum number of requests
            df_filtered = df[df['Name'] != 'SERP_KEY_5']

            if df_filtered.empty:
                return None

            best_key_name = df_filtered.loc[df_filtered['Status'].idxmax(), 'Name']
            best_key = df_filtered.loc[df_filtered['Status'].idxmax(), 'API Key']
            return best_key_name, best_key

        except Exception as e:
            print(f"Error while getting the best API key: {str(e)}")
            return None

    def stop_scheduler(self):
        """Stops the scheduler"""
        self.scheduler.shutdown()

########## Example usage ##########

# api_key_manager = APIKeyManager()

# if __name__ == "__main__":
#     try:
#         # Get the best available key
#         best_key_name, best_key = api_key_manager.get_best_api_key()
#         if best_key:
#             print(f"The best available API key: {best_key_name}: {best_key}")
#         else:
#             print("Failed to find an available API key")

#     except KeyboardInterrupt:
#         print("Shutting down...")
#         api_key_manager.stop_scheduler()
