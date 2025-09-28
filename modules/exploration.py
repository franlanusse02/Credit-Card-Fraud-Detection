# using https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud as a dataset

import pandas as pd

df = pd.read_csv('./downloads/creditcard.csv')


# debug prints to check database integrity:

# print('First 5 columns: \n{}'.format(df.head(5)))
# print('Column names: \n{}'.format(list(df.columns)))
# for column in df:
#     print('{} column: {} data type'.format(column, df[column].dtype))
#     print('{} column: {} missing values'.format(column, df[column].isna().sum()))

print(df.head())




