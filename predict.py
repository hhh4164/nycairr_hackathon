from datetime import datetime, timedelta
import pandas as pd

from train.process import download_day, process_day

#currently data a day behind
def get_predictions():
    day = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-$m-$d")

    try:
        download_day(day)
        processed_df = process_day(day)
    except Exception as e:
        print(f"Error processing data for {day}: {e}")
        processed_df = None

