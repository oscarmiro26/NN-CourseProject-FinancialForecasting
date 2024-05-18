import pandas as pd
import os

file_path = '~/Desktop/NeuralNetworks/NNProject/'
file_name = 'M3C.xls'
sheet = 'M3Month'

df = pd.read_excel(os.path.join(file_path, file_name), sheet_name=sheet)

csv_file = 'M3C_Monthly.csv'
df.to_csv(os.path.join(file_path, csv_file), index=False)
