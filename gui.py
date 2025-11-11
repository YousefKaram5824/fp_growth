import flet as ft
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plot
import pandas as pand
from fpg import FPTree, mine_frequent_itemsets, generate_association_rules
from collections import defaultdict as dict, Counter as count
import base64
from io import BytesIO


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
    rules_table = create_styled_datatable(
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

    file_picker = ft.FilePicker(on_result=pick_file_result)
    page.overlay.append(file_picker)

    def select_file(e):
        file_picker.pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["xlsx"],
        )

    def run_algorithm(e):
        if not file_path.value:
            snack = ft.SnackBar(ft.Text("Please select an Excel file."))
            page.overlay.append(snack)
            snack.open = True
            page.update()
            return

        if (
            min_sup_field.value is None
            or min_conf_field.value is None
            or min_sup_field.value == ""
            or min_conf_field.value == ""
        ):
            snack = ft.SnackBar(
                ft.Text("Please enter minimum support and confidence values.")
            )
            page.overlay.append(snack)
            snack.open = True
            page.update()
            return

        try:
            min_sup = float(min_sup_field.value) / 100
            min_conf = float(min_conf_field.value) / 100
        except ValueError:
            snack = ft.SnackBar(
                ft.Text("Invalid minimum support or confidence values.")
            )
            page.overlay.append(snack)
            snack.open = True
            page.update()
            return

        try:
            df = pand.read_excel(file_path.value)
        except Exception as e:
            snack = ft.SnackBar(ft.Text(f"Error reading file: {str(e)}"))
            page.overlay.append(snack)
            snack.open = True
            page.update()
            return

        if "items" in df.columns:
            transactions = [list(dict.fromkeys(row["items"].split(","))) for _, row in df.iterrows()]
        else:
            transactions = []
            for _, row in df.iterrows():
                trans = [item for item in row.values if pand.notna(item)]
                transactions.append(list(dict.fromkeys(trans)))

        if not transactions:
            snack = ft.SnackBar(ft.Text("No transactions found in the file."))
            page.overlay.append(snack)
            snack.open = True
            page.update()
            return

        total_transactions = len(transactions)
        min_sup = int(min_sup * total_transactions)

        item_counts = count()
        for transaction in transactions:
            for item in set(transaction):
                item_counts[item] += 1
        frequent_items = {
            item: count for item, count in item_counts.items() if count >= min_sup
        }
        sorted_frequent_items = sorted(
            frequent_items.items(), key=lambda x: (-x[1], x[0])
        )
        item_order = {item: idx for idx, (item, _) in enumerate(sorted_frequent_items)}

        sorted_transactions = []
        for transaction in transactions:
            sorted_trans = [
                item for item in sorted_frequent_items if item[0] in transaction
            ]
            if sorted_trans:
                sorted_transactions.append(list(dict.fromkeys([item[0] for item in sorted_trans])))

        tree = FPTree()
        tree.build_from_transactions(sorted_transactions, min_sup)
        frequent_itemsets = mine_frequent_itemsets(tree, min_sup)
        Lk = dict(list)
        for itemset, support in frequent_itemsets:
            Lk[len(itemset)].append((itemset, support))
        for k in Lk:
            Lk[k].sort(key=lambda x: tuple(sorted(x[0])))
        rules = generate_association_rules(frequent_itemsets, transactions, min_conf)

        visualize_tree(tree)
        populate_frequent_items(sorted_frequent_items)
        populate_rearranged_dataset(sorted_transactions)
        populate_itemsets(Lk)
        populate_rules(rules)

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
        if frequent_items_table.rows is not None:
            frequent_items_table.rows.clear()
        else:
            frequent_items_table.rows = []
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
        if rearranged_dataset_table.rows is not None:
            rearranged_dataset_table.rows.clear()
        else:
            rearranged_dataset_table.rows = []
        for idx, transaction in enumerate(sorted_transactions, start=1):
            rearranged_dataset_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(idx))),
                        ft.DataCell(ft.Text(str(transaction))),
                    ]
                )
            )

    def populate_itemsets(Lk):
        if itemsets_table.rows is not None:
            itemsets_table.rows.clear()
        else:
            itemsets_table.rows = []
        for k in sorted(Lk.keys()):
            if k == 1:
                continue
            for itemset, support in Lk[k]:
                reversed_itemset = tuple(reversed(itemset))
                itemsets_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(f"L{k}")),
                            ft.DataCell(ft.Text(str(reversed_itemset))),
                            ft.DataCell(ft.Text(str(support))),
                        ]
                    )
                )

    def populate_rules(rules):
        if rules_table.rows is not None:
            rules_table.rows.clear()
        else:
            rules_table.rows = []
        for rule in rules:
            sorted_antecedent = list(dict.fromkeys(sorted(rule[0], key=lambda i: item_order.get(i, float('inf')))))
            sorted_consequent = list(dict.fromkeys(sorted(rule[1], key=lambda i: item_order.get(i, float('inf')))))
            rules_table.rows.append(
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
            ft.Text("Select Excel File:", size=20, weight=ft.FontWeight.BOLD),
            ft.Row([file_path, ft.ElevatedButton("Browse", on_click=select_file)]),
            min_sup_field,
            min_conf_field,
            ft.ElevatedButton("Run FP-Growth", on_click=run_algorithm),
            ft.Divider(),
            ft.Text("FP-Tree Visualization:", size=20, weight=ft.FontWeight.BOLD),
            tree_image,
            ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(
                                "Rearranged Dataset:",
                                size=20,
                                weight=ft.FontWeight.BOLD,
                            ),
                            rearranged_dataset_table,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                    ft.VerticalDivider(width=1),
                    ft.Column(
                        [
                            ft.Text(
                                "Frequent Items (L1):",
                                size=20,
                                weight=ft.FontWeight.BOLD,
                            ),
                            frequent_items_table,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                ],
                spacing=20,
                expand=True,
            ),
            ft.Divider(),
            ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(
                                "Frequent Itemsets (Lk):",
                                size=20,
                                weight=ft.FontWeight.BOLD,
                            ),
                            itemsets_table,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                    ft.VerticalDivider(width=1),
                    ft.Column(
                        [
                            ft.Text(
                                "Association Rules:", size=20, weight=ft.FontWeight.BOLD
                            ),
                            rules_table,
                        ],
                        spacing=10,
                        expand=True,
                    ),
                ],
                spacing=20,
                expand=True,
            ),
        ],
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        spacing=20,
    )

    page.add(main_content)


ft.app(target=main)
