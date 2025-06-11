import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import prepare_accident_data

pivot_data, y_real, data = prepare_accident_data('data/monatszahlen2505_verkehrsunfaelle_06_06_25.csv')

# Plot the data
pivot_data.plot(kind='line', logy=True, figsize=(12, 8), title='Historical Number of Accidents per Category (Logarithmic Scale) per Year')
plt.xlabel('Year')
plt.ylabel('Number of Accidents (Log Scale)')
plt.legend(title='Category')
plt.grid(True)
# save the plot as an image
plt.savefig('accidents_per_category.png')
plt.show()
plt.close()

print(pivot_data.head())
