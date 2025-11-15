import matplotlib, pandas, matplotlib.pyplot, base64, flet, itertools
from collections import defaultdict, Counter
from io import BytesIO

matplotlib.use("Agg")


class Node:
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, count=1):
        self.count += count


class Tree:
    def __init__(self):
        self.root = Node(None)
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
                new_node = Node(item, count, node)
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
            sorted_conditional_transactions, freq_items = preprocess_transactions(
                conditional_transactions, min_support
            )
            conditional_tree = Tree()
            conditional_tree.build_from_transactions(
                sorted_conditional_transactions, freq_items
            )
            itemsets.extend(
                get_frequent_itemsets(conditional_tree, min_support, new_prefix)
            )
    return itemsets


def get_association_rules(frequent_itemsets, total_transactions):
    rules = []
    supports = {
        tuple(sorted(itemset)): support for itemset, support in frequent_itemsets
    }
    for itemset, support_of_frequent_itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        support = support_of_frequent_itemset / total_transactions
        sorted_itemset = tuple(sorted(itemset))
        for i in range(1, len(itemset)):
            for j in itertools.combinations(sorted_itemset, i):
                if_part = tuple(j)
                then_part = tuple(sorted(set(sorted_itemset) - set(if_part)))
                confidence = support_of_frequent_itemset / supports[if_part]
                lift = confidence / (supports[then_part] / total_transactions)
                rules.append((if_part, then_part, support, confidence, lift))
    return rules


def create(columns, rows=None):
    return flet.DataTable(
        columns=columns,
        rows=rows or [],
        border=flet.border.all(1, flet.Colors.BLACK),
        border_radius=flet.border_radius.all(8),
        vertical_lines=flet.border.BorderSide(1, flet.Colors.BLACK),
        horizontal_lines=flet.border.BorderSide(1, flet.Colors.BLACK),
        heading_row_color=flet.Colors.with_opacity(0.05, flet.Colors.BLACK12),
        heading_row_height=45,
        data_row_max_height=55,
        column_spacing=20,
    )


def main(page: flet.Page):
    page.title = "FP-Growth(Task 1)"
    page.theme_mode = flet.ThemeMode.LIGHT
    page.window.maximized = True
    page.vertical_alignment = flet.MainAxisAlignment.START
    page.horizontal_alignment = flet.CrossAxisAlignment.CENTER
    item_order = {}

    file_path = flet.TextField(label="Path file", read_only=True, width=300)
    min_sup_field = flet.TextField(label="Minimum Support (%)", width=300)
    min_conf_field = flet.TextField(label="Minimum Confidence (%)", width=300)
    tree_image = flet.Image(width=600, height=400, src_base64="")
    frequent_items_table = create(
        [
            flet.DataColumn(flet.Text("Item")),
            flet.DataColumn(flet.Text("Support")),
        ]
    )
    rearranged_datasets_table = create(
        [
            flet.DataColumn(flet.Text("Transaction ID")),
            flet.DataColumn(flet.Text("Items")),
        ]
    )
    frequent_itemsets_table = create(
        [
            flet.DataColumn(flet.Text("Level")),
            flet.DataColumn(flet.Text("Itemset")),
            flet.DataColumn(flet.Text("Support")),
        ]
    )
    all_association_rules_table = create(
        [
            flet.DataColumn(flet.Text("If")),
            flet.DataColumn(flet.Text("Then")),
            flet.DataColumn(flet.Text("Support")),
            flet.DataColumn(flet.Text("Confidence")),
            flet.DataColumn(flet.Text("Lift")),
            flet.DataColumn(flet.Text("Correlation Type")),
        ]
    )
    strong_association_rules_table = create(
        [
            flet.DataColumn(flet.Text("If")),
            flet.DataColumn(flet.Text("Then")),
            flet.DataColumn(flet.Text("Support")),
            flet.DataColumn(flet.Text("Confidence")),
            flet.DataColumn(flet.Text("Lift")),
            flet.DataColumn(flet.Text("Correlation Type")),
        ]
    )

    def pick_file_result(e: flet.FilePickerResultEvent):
        if e.files:
            file_path.value = e.files[0].path
            page.update()

    def select_file(e):
        file_picker.pick_files(
            allow_multiple=False,
            file_type=flet.FilePickerFileType.CUSTOM,
            allowed_extensions=["xlsx"],
        )

    file_picker = flet.FilePicker(on_result=pick_file_result)
    page.overlay.append(file_picker)

    def run(e):
        # 1. Read the transactions table from an Excel file.
        data = pandas.read_excel(file_path.value)
        print(data.head())

        frequent_items_table.rows.clear()
        rearranged_datasets_table.rows.clear()
        frequent_itemsets_table.rows.clear()
        all_association_rules_table.rows.clear()
        strong_association_rules_table.rows.clear()
        min_support = float(min_sup_field.value) / 100
        min_confidence = float(min_conf_field.value) / 100

        # 2. Use horizontal data format like the data in section
        horizontal_data_format = [
            list(dict.fromkeys(row["items"].split(","))) for _, row in data.iterrows()
        ]
        print("Transactions:", horizontal_data_format)

        # Preprocess transactions to find frequent items
        total_transactions = len(horizontal_data_format)
        min_sup_count = min_support * total_transactions
        sorted_transactions, frequent_items = preprocess_transactions(
            horizontal_data_format, min_sup_count
        )
        print("Frequent items:", frequent_items)

        # Sort frequent items by support descendingly first then alphabetically if there are more than one item with same support
        sorted_frequent_items = sorted(
            frequent_items.items(), key=lambda item: (-item[1], item[0])
        )
        print("Sorted frequent items:", sorted_frequent_items)

        item_order.clear()
        for _, (item, support) in enumerate(sorted_frequent_items):
            item_order[item] = -support

        # 3. Represent the FP-tree for your data
        tree = Tree()
        tree.build_from_transactions(sorted_transactions, frequent_items)

        # 4. Generate all the frequent item sets.
        frequent_itemsets = get_frequent_itemsets(tree, min_sup_count)
        print("Frequent itemsets:", frequent_itemsets)

        # 9. Generate all the possible frequent item sets (L1, L2, ....LK).
        Lks = defaultdict(list)
        for itemset, support in frequent_itemsets:
            Lks[len(itemset)].append((itemset, support))
        print("Frequent itemsets by levels:", dict(Lks))

        # 5. Represent the frequent items in the form of association rules.
        all_association_rules = get_association_rules(
            frequent_itemsets, total_transactions
        )
        print("All possible association rules:", all_association_rules)

        # 6. Extract the strong rules.
        # 7. Calculate the dependencies between the items (lift).
        strong_association_rules = [
            rule for rule in all_association_rules if rule[3] >= min_confidence
        ]
        print("Strong association rules", strong_association_rules)

        draw_tree(tree)
        show_frequent_items(sorted_frequent_items)
        show_rearranged_datasets(sorted_transactions)
        show_frequent_itemsets(Lks)
        show_all_association_rules(all_association_rules)
        show_strong_association_rules(strong_association_rules)
        page.update()

    def draw_tree(tree):
        fig, ax = matplotlib.pyplot.subplots(figsize=(6, 4))
        positions, edges = tree.get_tree_nodes()
        if not positions:
            ax.text(
                0.5, 0.5, "No tree to visualize", ha="center", va="center", fontsize=12
            )
            ax.axis("off")
        elif positions:
            xs = [p["x"] for p in positions]
            ys = [p["y"] for p in positions]
            ax.scatter(xs, ys, s=800, c="lightblue", edgecolors="black", zorder=2)
            for p in positions:
                ax.annotate(
                    f"{p['item']}:{p['count']}",
                    (p["x"], p["y"]),
                    ha="center",
                    va="center",
                    zorder=3,
                )
            for e in edges:
                dx = e["to"][0] - e["from"][0]
                dy = e["to"][1] - e["from"][1]
                ax.arrow(
                    e["from"][0],
                    e["from"][1],
                    dx,
                    dy,
                    head_width=15,
                    head_length=20,
                    fc="black",
                    ec="black",
                    zorder=1,
                )
            min_x = min(p["x"] for p in positions)
            max_x = max(p["x"] for p in positions)
            min_y = min(p["y"] for p in positions)
            max_y = max(p["y"] for p in positions)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x + 200
            height = max_y - min_y + 200
            ax.set_xlim(center_x - width / 2, center_x + width / 2)
            ax.set_ylim(center_y - height / 2, center_y + height / 2)
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        tree_image.src_base64 = img_base64
        matplotlib.pyplot.close(fig)

    def show_frequent_items(sorted_frequent_items):
        for item, support in sorted_frequent_items:
            frequent_items_table.rows.append(
                flet.DataRow(
                    cells=[
                        flet.DataCell(flet.Text(str(item))),
                        flet.DataCell(flet.Text(str(support))),
                    ]
                )
            )

    def show_rearranged_datasets(sorted_transactions):
        for idx, transaction in enumerate(sorted_transactions, start=1):
            rearranged_datasets_table.rows.append(
                flet.DataRow(
                    cells=[
                        flet.DataCell(flet.Text(str(idx))),
                        flet.DataCell(flet.Text(str(transaction))),
                    ]
                )
            )

    def show_frequent_itemsets(Lks):
        for k in sorted(Lks.keys()):
            if k == 1:
                continue
            for itemset, support in Lks[k]:
                sorted_itemset = sorted(itemset, key=lambda x: item_order.get(x, 0))
                frequent_itemsets_table.rows.append(
                    flet.DataRow(
                        cells=[
                            flet.DataCell(flet.Text(f"L{k}")),
                            flet.DataCell(flet.Text(str(tuple(sorted_itemset)))),
                            flet.DataCell(flet.Text(str(support))),
                        ]
                    )
                )

    def show_all_association_rules(all_association_rules):
        for rule in all_association_rules:
            lift = rule[4]
            if lift > 1:
                correlation_type = "Positive Correlation"
            elif lift < 1:
                correlation_type = "Negative Correlation"
            else:
                correlation_type = "No Relation"
            all_association_rules_table.rows.append(
                flet.DataRow(
                    cells=[
                        flet.DataCell(
                            flet.Text(
                                str(sorted(rule[0], key=lambda x: item_order.get(x, 0)))
                            )
                        ),
                        flet.DataCell(
                            flet.Text(
                                str(sorted(rule[1], key=lambda x: item_order.get(x, 0)))
                            )
                        ),
                        flet.DataCell(flet.Text(f"{rule[2]:.2f}")),
                        flet.DataCell(flet.Text(f"{rule[3]:.2f}")),
                        flet.DataCell(flet.Text(f"{rule[4]:.2f}")),
                        flet.DataCell(flet.Text(correlation_type)),
                    ]
                )
            )

    def show_strong_association_rules(strong_association_rules):
        for rule in strong_association_rules:
            lift = rule[4]
            if lift > 1:
                correlation_type = "Positive Correlation"
            elif lift < 1:
                correlation_type = "Negative Correlation"
            else:
                correlation_type = "No Relation"
            strong_association_rules_table.rows.append(
                flet.DataRow(
                    cells=[
                        flet.DataCell(
                            flet.Text(
                                str(sorted(rule[0], key=lambda x: item_order.get(x, 0)))
                            )
                        ),
                        flet.DataCell(
                            flet.Text(
                                str(sorted(rule[1], key=lambda x: item_order.get(x, 0)))
                            )
                        ),
                        flet.DataCell(flet.Text(f"{rule[2]:.2f}")),
                        flet.DataCell(flet.Text(f"{rule[3]:.2f}")),
                        flet.DataCell(flet.Text(f"{rule[4]:.2f}")),
                        flet.DataCell(flet.Text(correlation_type)),
                    ]
                )
            )

    main_content = flet.Column(
        [
            flet.Row(
                [
                    file_path,
                    flet.ElevatedButton("Select File", on_click=select_file),
                    min_sup_field,
                    min_conf_field,
                    flet.ElevatedButton("Run", on_click=run),
                ],
                alignment=flet.MainAxisAlignment.CENTER,
            ),
            flet.Divider(),
            flet.Row(
                [
                    flet.Text("FP-Tree:"),
                    tree_image,
                ],
                alignment=flet.MainAxisAlignment.CENTER,
            ),
            flet.Divider(),
            flet.Row(
                [
                    flet.Column(
                        [
                            flet.Text("Rearranged Datasets:"),
                            rearranged_datasets_table,
                        ]
                    ),
                    flet.Column(
                        [
                            flet.Text("Frequent Items (L1):"),
                            frequent_items_table,
                        ]
                    ),
                    flet.Column(
                        [
                            flet.Text("Frequent Itemsets:"),
                            frequent_itemsets_table,
                        ]
                    ),
                ],
                vertical_alignment=flet.CrossAxisAlignment.START,
                alignment=flet.MainAxisAlignment.CENTER,
                spacing=20,
            ),
            flet.Row(
                [
                    flet.Column(
                        [
                            flet.Text("All Possible Association Rules:"),
                            all_association_rules_table,
                        ]
                    ),
                    flet.Column(
                        [
                            flet.Text("Strong Association Rules:"),
                            strong_association_rules_table,
                        ]
                    ),
                ],
                vertical_alignment=flet.CrossAxisAlignment.START,
                alignment=flet.MainAxisAlignment.CENTER,
                spacing=20,
            ),
        ],
        scroll=flet.ScrollMode.AUTO,
        expand=True,
        spacing=20,
    )

    page.add(main_content)


flet.app(target=main)
