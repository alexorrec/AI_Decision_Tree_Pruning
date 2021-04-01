import numpy as np
from collections import Counter
import node as N


def print_tree(node, level = 0):
    if node.answer != '':
        print('\t'*level, node.answer)
        return
    print(' '*level, node.label)
    for value, n in node.children:
        print(' '*(level+1), value)
        print_tree(n, level+2)


def get_entropy(df_target):
    elements, counts = np.unique(df_target, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


def information_gain(df, column, target):
    target_ent = get_entropy(df[target])
    values, counts = np.unique(df[column], return_counts=True)

    w_ent = 0
    for i in range(len(values)):
        w_ent += counts[i] / np.sum(counts) * get_entropy(df.where(df[column] == values[i]).dropna()[target])
    return target_ent - w_ent


def plurality_value(examples):
    return Counter(examples).most_common(1)[0][0]


def build_tree(df, attributes, target, parent=None):
    if df.empty:
        node = N.Node('')
        node.answer = plurality_value(df[target])
        return node

    elif len(np.unique(df[target])) <= 1:  # Tutti gli esempi sono uguali ritorna quella classificazione
        node = N.Node('')
        node.answer = np.unique(df[target])[0]
        return node

    elif len(attributes) == 0:
        node = N.Node('')
        node.answer = plurality_value(parent)
        return node

    else:
        gain_values = []
        for i in df.columns[:-1]:
            gain_values.append(information_gain(df, i, target))

        best_split_index = np.argmax(gain_values)
        best_split = df.columns[best_split_index]
        tree = N.Node(best_split)

        new_attributes = [i for i in attributes if i != best_split]
        new_parent = df[target]

        for value in np.unique(df[best_split]):
            min_df = df.where(df[best_split] == value).dropna()
            sub_tree = build_tree(min_df, new_attributes, target, new_parent)
            tree.children.append((value, sub_tree))

        return tree


def predict(node, test):
    if len(node.children) == 0:
        return node.answer

    else:
        attr = test[node.label]

        for i in range(len(node.children)):
            if attr == node.children[i][0]:
                return predict(node.children[i][1], test)
            else:
                #da implementare a seguito di pruning
                pass


def accuracy(tree, tests, target):

    n_good_predicts = 0

    for i in range(len(tests)):
        if predict(tree, tests.iloc[i]) == tests[target][i]:
            n_good_predicts += 1

    return n_good_predicts/len(tests)
