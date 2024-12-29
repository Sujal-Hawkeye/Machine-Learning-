import pandas as pd
df = pd.read_csv(r"D:\onedrive\Documents\stock_data.csv")
print(df)
df.loc[4, 'price'] = 600
df.loc[3, 'eps'] = 34
df.loc[1, 'people'] = 'SamWalton'
df.to_csv(r"D:\onedrive\Documents\stock_data.csv", index=False)
print(df)
