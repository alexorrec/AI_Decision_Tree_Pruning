import dtree_helper as dtree
import pandas as pd

# Mushroom dataset
'''
df = pd.read_csv('/Users/alessandrocerro/PycharmProjects/AI_Decision_Tree_Pruning/datasets/mushy_data.csv', sep=';')
df.columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
              'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
              'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat', 'class']
'''
# Nursery Dataset
df = pd.read_csv('/Users/alessandrocerro/PycharmProjects/AI_Decision_Tree_Pruning/datasets/nursery.csv')
df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df)

target = 'class'
attributes = df.columns[:-1]

train_ = df.iloc[0:round(len(df) * 70 / 100)]
test_ = df.iloc[round(len(df) * 70 / 100):].reset_index(drop=True)

tree = dtree.build_tree(train_, attributes, target)

x = dtree.accuracy(tree, test_, target)
print(x)
