class Node:
    def __init__(self, label):
        self.label = label
        self.children = []
        self.answer = ''

        self.isPruned = False
