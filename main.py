import dtree_helper as dtree
import pandas as pd

# Mushroom dataset

df = pd.read_csv('./datasets/mushy_data.csv', sep=';')
df.columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
              'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
              'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat', 'class']
'''

# Nursery Dataset
df = pd.read_csv('./datasets/nursery.csv')
df.columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
'''
# Randomizzo le righe, in alcuni sets sono ordinate per un qualche attributo
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df)

# MAIN
train_set, test_set, val_set = dtree.split_df(df)

# Il target si trova nell'ultima colonna di entrambi i datasets...
attributes = df.columns[:-1]
target = df.columns[-1]

print('Inizio a costruire l\'albero...')
# Learning
tree = dtree.build_tree(train_set, attributes, target)
# Reimposto la radice
tree.is_root = True
tree.is_internal = False

# Funzione di stampa, sconsiglio l'uso sul Dataset Nursery (sono circa 1000 nodi...)
# dtree.print_tree(tree)
lenght = dtree.node_measure(tree)
print(f'Quantitativo di nodi pre_pruning: {lenght}')
accuracy = dtree.accuracy(tree, test_set, target)
print(f'Accuracy prima del pruning: {accuracy}')

print()

print('Inizio il pruning...')
dtree.prune(tree, tree, val_set, target)

# dtree.print_tree(tree)
lenght = dtree.node_measure(tree)
print(f'Quantitativo di nodi pre_pruning: {lenght}')
accuracy_post = dtree.accuracy(tree, test_set, target)
print(f'Accuracy dopo pruning: {accuracy_post}')
