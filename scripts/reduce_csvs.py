import pandas as pd
import numpy as np
from pathlib import Path 
import os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='The folder with each CSV file', default='novideo_data/CSV_files/')
    parser.add_argument('--dest_folder', help='The folder to save reduced CSV files', default='novideo_data/CSV_files/Reduced_CSVS/')
    parser.add_argument('-r', '--remove', help="The number of rows to remove from the CSV", type=int, default=5)
    args = parser.parse_args()
    
    # CONFIG
    csvs = Path(args.folder).glob('*.csv')


    # Load and remove rows
    for i, csv in enumerate(csvs):
        df = pd.read_csv(csv)
        df = df[df.index % args.remove == 0]
        df.to_csv(args.dest_folder + csv.name, index=False)
        print(f"✅ Saved reduced {csv} to", args.dest_folder + csv.name)


