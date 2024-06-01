import pandas as pd
import os

file_path = 'Home/RUG/Neural Networks/NN-CourseProject-FinancialForecasting'
file_name = 'M3C.xls'
sheet = 'M3Month'

df = pd.read_excel(os.path.join(file_path, file_name), sheet_name=sheet)

df['Category'] = df['Category'].str.strip()

df_filtered = df[df['Category'] == 'FINANCE']

csv_file_filtered = 'M3C_Monthly_FINANCE.csv'

df_filtered.to_csv(os.path.join(file_path, csv_file_filtered), index=False)