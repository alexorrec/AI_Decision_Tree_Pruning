import numpy as np
from collections import Counter
import node as N


# 70% per training, 15% per testing, 15% per validation
def split_df(df):
    n = round(len(df) * 70 / 100)
    n_ = round(len(df) * 15 / 100)
    train_set = df[0: n - 1].reset_index(drop=True)
    test_set = df[n: n + n_ - 1].reset_index(drop=True)
    val_set = df[n + n_ :].reset_index(drop=True)

    return train_set, test_set, val_set


def print_tree(node, level=0):
    if node.answer != '':
        print('\t' * level, node.answer)
        return
    if not node.is_pruned:
        print(' ' * level, node.label)
    for value, n in node.children:
        print(' ' * (level + 1), value)
        print_tree(n, level + 2)


def get_entropy(column):
    counts = Counter(column)

    entropy = 0
    for i in counts.items():
        entropy += i[1]/len(column)*np.log2(i[1]/len(column))
    return -entropy


def information_gain(df, column, target):
    target_ent = get_entropy(df[target])
    values, counts = np.unique(df[column], return_counts=True)

    ent_ = 0
    for i in range(len(values)):
        ent_ += counts[i]/np.sum(counts)*get_entropy(df.where(df[column] == values[i]).dropna()[target])
    return target_ent - ent_


def plurality_value(examples):
    return Counter(examples).most_common(1)[0][0]


def importance(df, target):
    gain_values = []
    for i in df.columns[:-1]:
        gain_values.append(information_gain(df, i, target))

    best_split_index = np.argmax(gain_values)
    best_split = df.columns[best_split_index]

    return best_split


def build_tree(df, attributes, target, parent=None):
    if df.empty:
        node = N.Node('')
        node.is_leaf = True
        node.answer = plurality_value(df[target])
        return node

    elif len(np.unique(df[target])) <= 1:
        node = N.Node('')
        node.is_leaf = True
        node.answer = np.unique(df[target])[0]
        return node

    elif len(attributes) == 0:
        node = N.Node('')
        node.is_leaf = True
        node.answer = plurality_value(parent)
        return node

    else:
        best_split = importance(df, target)
        tree = N.Node(best_split)
        tree.is_internal = True

        new_attributes = [i for i in attributes if i != best_split]
        new_parent = df[target]

        for value in np.unique(df[best_split]):
            min_df = df.where(df[best_split] == value).dropna()
            sub_tree = build_tree(min_df, new_attributes, target, new_parent)
            tree.children.append((value, sub_tree))

        return tree


def predict(node, test):
    if len(node.children) == 0 or node.is_pruned:
        return node.answer
    else:
        attr = test[node.label]
        for i in range(len(node.children)):
            if attr == node.children[i][0]:
                return predict(node.children[i][1], test)


def accuracy(tree, tests, target):
    n_good_predicts = 0
    for i in range(len(tests)):
        if predict(tree, tests.iloc[i]) == tests[target][i]:
            n_good_predicts += 1
    return n_good_predicts / len(tests)


def prune(node, tree, val_set, target, modal=[]):
    if not node.is_leaf:
        for i in range(len(node.children)):
            prune(node.children[i][1], tree, val_set, target, modal)
            if node.is_root:
                modal.clear()
    elif node.is_leaf:
        modal.append(node.answer)
        return

    if node.is_internal:
        prior_acc = accuracy(tree, val_set, target)
        node.is_pruned = True
        node.answer = plurality_value(modal)

        if accuracy(tree, val_set, target) >= prior_acc:
            modal.clear()
            modal.append(node.answer)
            return
        else:
            node.is_pruned = False
            node.answer = ''
            return
