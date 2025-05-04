import pandas as pd
import numpy as np
import os.path as path
import os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='The folder with each CSV file', default='novideo_data/CSV_files/')
    parser.add_argument('-r', '--remove', help="The number of rows to remove from the CSV", type=int, default=15)
    args = parser.parse_args()
    
    # CONFIG
    csvs = [os.path.join(args.folder, file) for file in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, file))]


    # Load and remove rows
    for i, csv in enumerate(csvs):
        df = pd.read_csv(csv)
        df = df[df.index % args.remove == 0]
        df.to_csv("novideo_data/CSV_files/Cleaned_CSVS/" + os.path.basename(csv), index=False)
        


