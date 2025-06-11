import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepare_accident_data(csv_path, year_limit=2020):
    original_data = pd.read_csv(csv_path)
    # Remove data records after year_limit to create a consistent dataset for training your model
    data = original_data[original_data['JAHR'] <= year_limit]

    # Optionally, get data for the next year (not used in pivot)
    data_next_year = original_data[original_data['JAHR'] == year_limit + 1]

    # Filter the dataset to include only rows where 'MONAT' is 'Summe'
    sum_value_dataset = data[data['MONAT'] == 'Summe']
    sum_value_dataset = sum_value_dataset[sum_value_dataset['AUSPRAEGUNG'] == 'insgesamt']

    # Pivot the data to have years as rows and MONATSZAHL as columns
    pivot_data = sum_value_dataset.pivot(index='JAHR', columns='MONATSZAHL', values='WERT')
    # Convert the 'JAHR' column to a categorical type
    pivot_data.index = pivot_data.index.astype('category')

    target_month = '202101'
    data_2021 = original_data[original_data['JAHR'] == 2021]
    y_real = data_2021[data_2021['MONAT'] == target_month]['WERT'].values[0]

    # Filter the data for the first month of each year
    alkoholunfaelle_data = data[data['MONATSZAHL'] == 'Alkoholunfälle']
    first_month_data = alkoholunfaelle_data[alkoholunfaelle_data['MONAT'].str.endswith('01')]
    # Only intersted in insgesamt
    first_month_data = first_month_data[first_month_data['AUSPRAEGUNG'] == 'insgesamt']

    return pivot_data, y_real, data

pivot_data, y_real, data = prepare_accident_data('data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv')
