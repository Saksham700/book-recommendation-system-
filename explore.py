import pandas as pd
df_c = pd.read_csv('chapters.csv')
df_i = pd.read_csv('interactions.csv', nrows=10)
with open('data_info.txt', 'w') as f:
    f.write('chapters.csv head:\n' + df_c.head().to_string() + '\n\n')
    f.write('chapters.csv columns:\n' + str(df_c.columns) + '\n\n')
    f.write('interactions.csv head:\n' + df_i.head().to_string() + '\n\n')
    f.write('interactions.csv columns:\n' + str(df_i.columns) + '\n')
