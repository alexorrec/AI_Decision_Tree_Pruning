class Node:
    def __init__(self, label):
        self.label = label
        self.children = []  # Contiene [value, Next_Nodo]
        self.answer = ''

        self.is_pruned = False

        self.is_root = False
        self.is_internal = False
        self.is_leaf = False
