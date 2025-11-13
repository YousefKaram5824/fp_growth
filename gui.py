import matplotlib, pandas, matplotlib.pyplot as plot, base64, flet as ft
from collections import defaultdict
from io import BytesIO
from fpg import (
    FPTree,
    get_frequent_itemsets,
    generate_association_rules,
    preprocess_transactions,
)

matplotlib.use("Agg")


def create_styled_datatable(columns, rows=None):
    return ft.DataTable(
        columns=columns,
        rows=rows or [],
        border=ft.border.all(1, ft.Colors.BLACK),
        border_radius=ft.border_radius.all(8),
        vertical_lines=ft.border.BorderSide(1, ft.Colors.BLACK),
        horizontal_lines=ft.border.BorderSide(1, ft.Colors.BLACK),
        heading_row_color=ft.Colors.with_opacity(0.05, ft.Colors.BLACK12),
        heading_row_height=45,
        data_row_max_height=55,
        column_spacing=20,
    )


def main(page: ft.Page):
    page.title = "FP-Growth(Task 1)"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.maximized = True
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Global variables for consistent ordering
    frequent_items = {}
    item_order = {}

    file_path = ft.TextField(label="Path file", read_only=True, width=300)
    min_sup_field = ft.TextField(label="Minimum Support (%)", width=300)
    min_conf_field = ft.TextField(label="Minimum Confidence (%)", width=300)
    tree_image = ft.Image(width=600, height=400, src_base64="")
    frequent_items_table = create_styled_datatable(
        [
            ft.DataColumn(ft.Text("Item")),
            ft.DataColumn(ft.Text("Support")),
        ]
    )
    rearranged_dataset_table = create_styled_datatable(
        [
            ft.DataColumn(ft.Text("Transaction ID")),
            ft.DataColumn(ft.Text("Items")),
        ]
    )
    itemsets_table = create_styled_datatable(
        [
            ft.DataColumn(ft.Text("Level")),
            ft.DataColumn(ft.Text("Itemset")),
            ft.DataColumn(ft.Text("Support")),
        ]
    )
    all_association_rules_table = create_styled_datatable(
        [
            ft.DataColumn(ft.Text("Antecedent")),
            ft.DataColumn(ft.Text("Consequent")),
            ft.DataColumn(ft.Text("Support")),
            ft.DataColumn(ft.Text("Confidence")),
            ft.DataColumn(ft.Text("Lift")),
        ]
    )
    strong_association_rules_table = create_styled_datatable(
        [
            ft.DataColumn(ft.Text("Antecedent")),
            ft.DataColumn(ft.Text("Consequent")),
            ft.DataColumn(ft.Text("Support")),
            ft.DataColumn(ft.Text("Confidence")),
            ft.DataColumn(ft.Text("Lift")),
        ]
    )

    def pick_file_result(e: ft.FilePickerResultEvent):
        if e.files:
            file_path.value = e.files[0].path
            page.update()

    def select_file(e):
        file_picker.pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["xlsx"],
        )

    file_picker = ft.FilePicker(on_result=pick_file_result)
    page.overlay.append(file_picker)

    def run_algorithm(e):
        frequent_items_table.rows.clear()
        rearranged_dataset_table.rows.clear()
        itemsets_table.rows.clear()
        all_association_rules_table.rows.clear()
        strong_association_rules_table.rows.clear()
        min_sup = float(min_sup_field.value) / 100
        min_conf = float(min_conf_field.value) / 100

        # 1. Read the transactions table from an Excel file.
        data = pandas.read_excel(file_path.value)
        print(data.head())

        # 2. Use horizontal data format like the data in section
        horizontal_data_format = [
            list(dict.fromkeys(row["items"].split(","))) for _, row in data.iterrows()
        ]
        print("Transactions:", horizontal_data_format)

        # Preprocess transactions to find frequent items
        total_transactions = len(horizontal_data_format)
        min_sup_count = min_sup * total_transactions
        sorted_transactions, frequent_items_local = preprocess_transactions(
            horizontal_data_format, min_sup_count
        )
        nonlocal frequent_items, item_order
        frequent_items = frequent_items_local
        print("Frequent items:", frequent_items)

        # Sort frequent items by support descendingly first then alphabetically if there are more than one item with same support
        sorted_frequent_items = sorted(
            frequent_items.items(), key=lambda item: (-item[1], item[0])
        )
        print("Sorted frequent items:", sorted_frequent_items)

        # Create global item order for consistent sorting
        item_order.clear()
        for i, (item, support) in enumerate(sorted_frequent_items):
            item_order[item] = -support

        # 3. Represent the FP-tree for your data
        tree = FPTree()
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
        all_association_rules = generate_association_rules(
            frequent_itemsets, total_transactions
        )
        print("All possible association rules:", all_association_rules)

        # 6. Extract the strong rules.
        # 7. Calculate the dependencies between the items (lift).
        strong_association_rules = [
            rule for rule in all_association_rules if rule[3] >= min_conf
        ]
        print("Strong association rules", strong_association_rules)

        visualize_tree(tree)
        populate_frequent_items(sorted_frequent_items)
        populate_rearranged_dataset(sorted_transactions)
        populate_itemsets(Lks)
        populate_all_association_rules(all_association_rules)
        populate_strong_association_rules(strong_association_rules)
        page.update()

    def visualize_tree(tree):
        fig, ax = plot.subplots(figsize=(6, 4))
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
        plot.close(fig)

    def populate_frequent_items(sorted_frequent_items):
        for item, support in sorted_frequent_items:
            frequent_items_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(item))),
                        ft.DataCell(ft.Text(str(support))),
                    ]
                )
            )

    def populate_rearranged_dataset(sorted_transactions):
        for idx, transaction in enumerate(sorted_transactions, start=1):
            rearranged_dataset_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(idx))),
                        ft.DataCell(ft.Text(str(transaction))),
                    ]
                )
            )

    def populate_itemsets(Lks):
        for k in sorted(Lks.keys()):
            if k == 1:
                continue
            for itemset, support in Lks[k]:
                sorted_itemset = sorted(itemset, key=lambda x: item_order.get(x, 0))
                itemsets_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(f"L{k}")),
                            ft.DataCell(ft.Text(str(tuple(sorted_itemset)))),
                            ft.DataCell(ft.Text(str(support))),
                        ]
                    )
                )

    def populate_all_association_rules(all_association_rules):
        for rule in all_association_rules:
            sorted_antecedent = sorted(rule[0], key=lambda x: item_order.get(x, 0))
            sorted_consequent = sorted(rule[1], key=lambda x: item_order.get(x, 0))
            all_association_rules_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(sorted_antecedent))),
                        ft.DataCell(ft.Text(str(sorted_consequent))),
                        ft.DataCell(ft.Text(f"{rule[2]:.2f}")),
                        ft.DataCell(ft.Text(f"{rule[3]:.2f}")),
                        ft.DataCell(ft.Text(f"{rule[4]:.2f}")),
                    ]
                )
            )

    def populate_strong_association_rules(strong_association_rules):
        for rule in strong_association_rules:
            sorted_antecedent = sorted(rule[0], key=lambda x: item_order.get(x, 0))
            sorted_consequent = sorted(rule[1], key=lambda x: item_order.get(x, 0))
            strong_association_rules_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(sorted_antecedent))),
                        ft.DataCell(ft.Text(str(sorted_consequent))),
                        ft.DataCell(ft.Text(f"{rule[2]:.2f}")),
                        ft.DataCell(ft.Text(f"{rule[3]:.2f}")),
                        ft.DataCell(ft.Text(f"{rule[4]:.2f}")),
                    ]
                )
            )

    main_content = ft.Column(
        [
            ft.Row([file_path, ft.ElevatedButton("Select File", on_click=select_file)]),
            ft.Row(
                [
                    min_sup_field,
                    min_conf_field,
                    ft.ElevatedButton("Run FP-Growth", on_click=run_algorithm),
                ]
            ),
            ft.Divider(),
            ft.Text("FP-Tree:", size=16, weight=ft.FontWeight.BOLD),
            tree_image,
            ft.Text(
                "Rearranged Dataset:",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            rearranged_dataset_table,
            ft.Text(
                "Frequent Items (L1):",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            frequent_items_table,
            ft.Text(
                "Frequent Itemsets:",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            itemsets_table,
            ft.Text(
                "All Possible Association Rules:",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            all_association_rules_table,
            ft.Text(
                "Strong Association Rules:",
                size=16,
                weight=ft.FontWeight.BOLD,
            ),
            strong_association_rules_table,
        ],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=20,
    )

    page.add(main_content)


ft.app(target=main)
