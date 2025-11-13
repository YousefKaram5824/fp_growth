from collections import defaultdict, Counter
import itertools


def preprocess_transactions(transactions, min_support):
    item_counts = Counter()
    for transaction in transactions:
        for item in set(transaction):
            item_counts[item] += 1
    frequent_items = {
        item: support for item, support in item_counts.items() if support >= min_support
    }
    sorted_items = sorted(frequent_items.keys(), key=lambda x: (-frequent_items[x], x))
    sorted_transactions = []
    for transaction in transactions:
        sorted_trans = [item for item in sorted_items if item in transaction]
        if sorted_trans:
            sorted_transactions.append(sorted_trans)
    return sorted_transactions, frequent_items


class FPNode:
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, count=1):
        self.count += count


class FPTree:
    def __init__(self):
        self.root = FPNode(None)
        self.header_table = defaultdict(list)
        self.item_supports = {}

    def get_tree_nodes(
        self, node=None, x=0, y=1000, level=0, positions=None, edges=None
    ):
        if positions is None:
            positions = []
        if edges is None:
            edges = []
        if node is None:
            node = self.root
        if node.item is None:  # Root node
            positions.append(
                {"item": "null", "count": 0, "x": x, "y": y, "level": level}
            )
        else:
            positions.append(
                {"item": node.item, "count": node.count, "x": x, "y": y, "level": level}
            )
        sorted_children = sorted(
            node.children.keys(),
            key=lambda x: self.item_supports.get(x, 0),
            reverse=True,
        )
        child_x = x - 100 * (len(node.children) - 1) / 2
        for child_item in sorted_children:
            child_node = node.children[child_item]
            child_pos = {
                "item": child_node.item,
                "count": child_node.count,
                "x": int(child_x),
                "y": y - 100,
                "level": level + 1,
            }
            edges.append({"from": (x, y), "to": (child_pos["x"], child_pos["y"])})
            self.get_tree_nodes(
                child_node, int(child_x), y - 100, level + 1, positions, edges
            )
            child_x += 100
        return positions, edges

    def insert_transaction(self, transaction, count=1):
        node = self.root
        for item in transaction:
            if item in node.children:
                node = node.children[item]
                node.increment(count)
            else:
                new_node = FPNode(item, count, node)
                node.children[item] = new_node
                self.header_table[item].append(new_node)
                node = new_node

    def build_from_transactions(self, sorted_transactions, frequent_items):
        self.item_supports = frequent_items
        for transaction in sorted_transactions:
            self.insert_transaction(transaction)
        for item in self.header_table:
            nodes = self.header_table[item]
            for i in range(len(nodes) - 1):
                nodes[i].next = nodes[i + 1]


def get_frequent_itemsets(tree, min_support, prefix=()):
    itemsets = []
    for item in tree.header_table.keys():
        new_prefix = prefix + (item,)
        itemsets.append(
            (new_prefix, sum(node.count for node in tree.header_table[item]))
        )
        conditional_transactions = []
        node = tree.header_table[item][0]
        while node:
            path = []
            parent = node.parent
            while parent and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                conditional_transactions.extend([path] * node.count)
            node = node.next
        if conditional_transactions:
            sorted_cond_trans, freq_items = preprocess_transactions(
                conditional_transactions, min_support
            )
            conditional_tree = FPTree()
            conditional_tree.build_from_transactions(sorted_cond_trans, freq_items)
            itemsets.extend(
                get_frequent_itemsets(conditional_tree, min_support, new_prefix)
            )
    return itemsets


def generate_association_rules(frequent_itemsets, total_transactions):
    rules = []
    itemset_supports = {
        frozenset(itemset): support for itemset, support in frequent_itemsets
    }

    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = frozenset(set(itemset) - set(antecedent))
                    if (
                        antecedent in itemset_supports
                        and itemset_supports[antecedent] > 0
                        and consequent in itemset_supports
                    ):
                        confidence = support / itemset_supports[antecedent]
                        lift = confidence / (
                            itemset_supports[consequent] / total_transactions
                        )
                        rules.append(
                            (
                                antecedent,
                                consequent,
                                support / total_transactions,
                                confidence,
                                lift,
                            )
                        )
    return rules
