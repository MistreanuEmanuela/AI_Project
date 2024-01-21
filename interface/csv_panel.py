import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import csv


class CSVViewer:
    def __init__(self, root):
        self.root = root
        self.frame1 = None
        self.frame2 = None

    def configure_tree_columns(self, tree, header):
        tree["columns"] = header
        for col in header:
            tree.heading(col, text=col)
            tree.column(col, width=150, minwidth=50, stretch=tk.NO)

    def insert_csv_data(self, tree, csv_reader):
        for row in csv_reader:
            tree.insert("", "end", values=row)

    def adjust_tree_height(self, tree):
        tree_height = min(10, len(tree.get_children()))
        tree["height"] = tree_height

    def adjust_column_widths(self, tree, header):
        for col in header:
            tree.column(col, width=tkFont.Font().measure(col) + 10)

    def display_csv_data(self, tree, file_path):
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            tree.delete(*tree.get_children())

            self.configure_tree_columns(tree, header)
            self.insert_csv_data(tree, csv_reader)
            self.adjust_tree_height(tree)
            self.adjust_column_widths(tree, header)

    def setup_csv_section(self, path):
        self.frame1 = ttk.Frame(self.root)
        self.frame1.place(x=20, y=150, width=700, height=250)
        self.frame1.pack_propagate(False)

        tree = ttk.Treeview(self.frame1, show="headings")
        vsb = ttk.Scrollbar(self.frame1, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.frame1, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')

        tree.place(x=0, y=0, width=700, height=250)

        self.display_csv_data(tree, path)

    def setup_csv_preprocessing_section(self, path):
        self.frame2 = ttk.Frame(self.root)
        self.frame2.place(x=20, y=430, width=700, height=250)
        self.frame2.pack_propagate(False)

        tree = ttk.Treeview(self.frame2, show="headings")
        vsb = ttk.Scrollbar(self.frame2, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.frame2, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')

        tree.place(x=0, y=0, width=700, height=250)

        self.display_csv_data(tree, path)

    def hide_csv(self):
        self.frame1.destroy()
        self.frame2.destroy()
