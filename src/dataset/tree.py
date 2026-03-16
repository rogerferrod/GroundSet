class Node:
    def __init__(self, name, level, parent, is_leaf=False, children=None):
        self.name = name.lower()
        self.level = level
        self.parent = parent
        self.children = children
        self.is_leaf = is_leaf

    def __str__(self):
        return f'{self.name} ({self.level})'

    def get_descendants(self):
        return Node._rec_descendants(self)

    def get_parent(self, level=0):
        if self.level == level:
            return self

        parent = self.parent
        if parent.level == level:
            return parent

        return parent.get_parent(level)

    @staticmethod
    def _rec_descendants(node):
        if node.is_leaf:
            return {node.name}

        descendants = {node.name}
        for child in node.children:
            descendants.update(Node._rec_descendants(child))

        return descendants


class Tree:
    def __init__(self, root):
        self.root = root

    def find(self, name):
        return Tree._rec_find(name, self.root)

    @staticmethod
    def _rec_find(name, node):
        if node is None:
            return None

        if node.name == name.lower():
            return node

        if not node.is_leaf:
            for child in node.children:
                found = Tree._rec_find(name, child)
                if found is not None:
                    return found


def load_tree(tree, level=0, parent=None):
    nodes = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            node = Node(k, level, parent)
            node.children = load_tree(v, level + 1, node)
            nodes.append(node)
    elif isinstance(tree, list):
        return [Node(x, level, parent, is_leaf=True) for x in tree]

    return nodes
