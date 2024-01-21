import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk
import PIL
import os
from interface.csv_panel import CSVViewer
from logical.new_instance_prediction import predict_dataset_1
from logical.new_instance_prediction2 import predict_dataset_2


class ViewData:
    def __init__(self, root):
        self.entry_widget = None
        self.csv_frame = None
        self.canvas = None
        self.text_label_data = None
        self.text_label_preproc = None

        self.diagrams = "diagrams"

        self.window = root
        self.setup_view_data()
        self.buttons()

        self.path1 = "./data/HH_Provider_Jan2024.csv"
        self.path_preprocessing1 = "./data/Preprocessing_HH.csv"

        self.path2 = "./data/HHCAHPS_Provider_Jan2024.csv"
        self.path_preprocessing2 = "./data/Preprocessing_HHCAHPS.csv"

        self.css_viewer = CSVViewer(self.window)

        self.css_viewer.setup_csv_section(self.path1)
        self.css_viewer.setup_csv_preprocessing_section(self.path_preprocessing1)

        self.id3_label = None

    def setup_view_data(self):
        self.canvas = tk.Canvas(self.window, bd=0, highlightthickness=0, relief='ridge')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.create_text(20, 20,
                                text="Selectează fișierul cu date:",
                                font=("Arial", 16, "bold"), anchor='nw')

        self.canvas.create_text(825, 20,
                                text="Încearcă o predicție",
                                font=("Arial", 16, "bold"), anchor='nw')

        self.canvas.create_text(20, 680,
                                text="Vizualizează grafice",
                                font=("Arial", 12, "bold"), anchor='nw')

        self.canvas.create_text(175, 681,
                                text="(din primul csv)",
                                font=("Arial", 10), anchor='nw')

        self.text_label_data = self.canvas.create_text(20, 120,
                                                       text="HH_Provider_Jan2024.csv",
                                                       font=("Arial", 12, "bold"), anchor='nw')

        self.text_label_preproc = self.canvas.create_text(20, 400,
                                                          text="Preprocessing_HH.csv",
                                                          font=("Arial", 12, "bold"), anchor='nw')

        self.text_label_col1 = self.canvas.create_text(760, 80,
                                                       text="- How often patients got better \n  at walking or moving around (%):",
                                                       font=("Arial", 13), anchor='nw')
        entry_x = 1050
        entry_y = 85
        self.entry_var1 = tk.StringVar()
        entry_widget = tk.Entry(self.canvas, textvariable=self.entry_var1, width=6, font=("Arial", 13), validate="key",
                                validatecommand=(self.canvas.register(self.validate_entry), "%P"))
        entry_widget.place(x=entry_x, y=entry_y)

        self.text_label_col2 = self.canvas.create_text(760, 135,
                                                       text="- How often patients got better \n  at getting in and out of bed (%):",
                                                       font=("Arial", 13), anchor='nw')

        entry_x = 1050
        entry_y = 140
        self.entry_var2 = tk.StringVar()
        entry_widget = tk.Entry(self.canvas, textvariable=self.entry_var2, width=6, font=("Arial", 13), validate="key",
                                validatecommand=(self.canvas.register(self.validate_entry), "%P"))
        entry_widget.place(x=entry_x, y=entry_y)

        self.text_label_col3 = self.canvas.create_text(760, 190,
                                                       text="- How often patients got better \n  at bathing (%):",
                                                       font=("Arial", 13), anchor='nw')

        entry_x = 1050
        entry_y = 195
        self.entry_var3 = tk.StringVar()
        entry_widget = tk.Entry(self.canvas, textvariable=self.entry_var3, width=6, font=("Arial", 13), validate="key",
                                validatecommand=(self.canvas.register(self.validate_entry), "%P"))
        entry_widget.place(x=entry_x, y=entry_y)

        self.text_label_col4 = self.canvas.create_text(760, 245,
                                                       text="- How often patients' breathing \n  improved (%):",
                                                       font=("Arial", 13), anchor='nw')

        entry_x = 1050
        entry_y = 250
        self.entry_var4 = tk.StringVar()
        entry_widget = tk.Entry(self.canvas, textvariable=self.entry_var4, width=6, font=("Arial", 13), validate="key",
                                validatecommand=(self.canvas.register(self.validate_entry), "%P"))
        entry_widget.place(x=entry_x, y=entry_y)

        self.text_label_col5 = self.canvas.create_text(760, 300,
                                                       text="- How often patients got better \n  at taking their drugs correctly by \n  mouth (%):",
                                                       font=("Arial", 13), anchor='nw')

        entry_x = 1050
        entry_y = 305
        self.entry_var5 = tk.StringVar()
        self.entry_widget = tk.Entry(self.canvas, textvariable=self.entry_var5, width=6, font=("Arial", 13),
                                     validate="key",
                                     validatecommand=(self.canvas.register(self.validate_entry), "%P"))
        self.entry_widget.place(x=entry_x, y=entry_y)

    def validate_entry(self, new_value):
        return len(new_value) <= 6

    def buttons(self):
        button_width = 15
        button_height = 2

        button_font = ("Arial", 12)

        button1 = tk.Button(self.window, text="HHS_Data1.csv",
                            command=lambda: self.on_button_data1(),
                            width=button_width, height=button_height, borderwidth=1, highlightthickness=0,
                            font=button_font, bg="#D9D9D9")
        self.canvas.create_window(20, 60, window=button1, anchor='nw')

        button2 = tk.Button(self.window, text="HHS_Data2.csv",
                            command=lambda: self.on_button_data2(),
                            width=button_width, height=button_height, borderwidth=1, highlightthickness=0,
                            font=button_font, bg="#D9D9D9")
        self.canvas.create_window(180, 60, window=button2, anchor='nw')

        button3 = tk.Button(self.window, text="Medii și mediane",
                            command=lambda: self.on_button_average_medians(),
                            width=button_width, height=1, borderwidth=1, highlightthickness=0,
                            font=button_font, bg="#D9D9D9")
        self.canvas.create_window(20, 710, window=button3, anchor='nw')

        button4 = tk.Button(self.window, text="Nr. inst. per clase",
                            command=lambda: self.on_button_nr_inst_classes(),
                            width=button_width, height=1, borderwidth=1, highlightthickness=0,
                            font=button_font, bg="#D9D9D9")
        self.canvas.create_window(200, 710, window=button4, anchor='nw')

        button5 = tk.Button(self.window, text="Statistici algoritmi\nde clasificare",
                            command=lambda: self.on_button_stats(),
                            width=18, height=3, borderwidth=1, highlightthickness=0,
                            font=button_font, bg="#D9D9D9")
        self.canvas.create_window(535, 680, window=button5, anchor='nw')

        button6 = tk.Button(self.window, text="Start",
                            command=lambda: self.on_button_click(),
                            width=10, height=2, borderwidth=1, highlightthickness=0,
                            font=button_font, bg="#D9D9D9")
        self.canvas.create_window(900, 380, window=button6, anchor='nw')

    def predictions_labels(self, id3_prediction, RN_continuous_prediction, RN_multiclass_prediction, bayes_prediction,
                           linear_regression_prediction, random_forest_prediction, ada_boost_prediction):

        if self.diagrams == "diagrams":
            self.label_algs_text = tk.Label(self.canvas,
                                            text="Predicții pentru target-ul\n'Quality of patient care star rating':",
                                            font=("Arial", 13, "bold"))
            self.canvas.create_window(800, 450, window=self.label_algs_text, anchor='nw')
        else:
            self.label_algs_text = tk.Label(self.canvas,
                                            text="Predicții pentru target-ul\n'HHCAHPS Survey Summary Star Rating':",
                                            font=("Arial", 13, "bold"))
            self.canvas.create_window(770, 450, window=self.label_algs_text, anchor='nw')

        self.id3_label = tk.Label(self.canvas, text=f"- Algoritmul ID3: {id3_prediction}", font=("Arial", 13))
        self.canvas.create_window(760, 510, window=self.id3_label, anchor='nw')

        self.rn_continuous_label = tk.Label(self.canvas,
                                            text=f"- Rețele Neuronale (valori continue): {RN_continuous_prediction}",
                                            font=("Arial", 13))
        self.canvas.create_window(760, 535, window=self.rn_continuous_label, anchor='nw')

        self.rn_multiclass_label = tk.Label(self.canvas,
                                            text=f"- Rețele Neuronale (clase multiple): {RN_multiclass_prediction}",
                                            font=("Arial", 13))
        self.canvas.create_window(760, 560, window=self.rn_multiclass_label, anchor='nw')

        self.bayes_label = tk.Label(self.canvas, text=f"- Algoritmul Bayes Naiv: {bayes_prediction}",
                                    font=("Arial", 13))
        self.canvas.create_window(760, 585, window=self.bayes_label, anchor='nw')

        self.linear_regression_label = tk.Label(self.canvas,
                                                text=f"- Regresia Liniară: {linear_regression_prediction}",
                                                font=("Arial", 13))
        self.canvas.create_window(760, 610, window=self.linear_regression_label, anchor='nw')

        self.rf_label = tk.Label(self.canvas, text=f"- Random Forest: {random_forest_prediction}",
                                 font=("Arial", 13))
        self.canvas.create_window(760, 635, window=self.rf_label, anchor='nw')

        self.ada_boost_label = tk.Label(self.canvas, text=f"- Algoritmul AdaBoost: {ada_boost_prediction}",
                                        font=("Arial", 13))
        self.canvas.create_window(760, 660, window=self.ada_boost_label, anchor='nw')

    def on_button_click(self):

        try:
            entered_value1 = float(self.entry_var1.get())
            entered_value2 = float(self.entry_var2.get())
            entered_value3 = float(self.entry_var3.get())
            entered_value4 = float(self.entry_var4.get())

            if self.diagrams != "diagrams2":
                entered_value5 = float(self.entry_var5.get())

                if entered_value1 and entered_value2 and entered_value3 and entered_value4 and entered_value5:
                    if 0.0 <= entered_value1 <= 100.0 and 0.0 <= entered_value2 <= 100.0 and 0.0 <= entered_value3 <= 100.0 and 0.0 <= entered_value4 <= 100.0 and 0.0 <= entered_value5 <= 100.0:

                        (id3_prediction, RN_continuous_prediction, RN_multiclass_prediction, bayes_prediction,
                         linear_regression_prediction, random_forest_prediction, ada_boost_prediction) \
                            = predict_dataset_1(entered_value1, entered_value2, entered_value3, entered_value4,
                                                entered_value5)
                        self.predictions_labels(id3_prediction, RN_continuous_prediction, RN_multiclass_prediction,
                                                bayes_prediction, linear_regression_prediction,
                                                random_forest_prediction, ada_boost_prediction)
                    else:
                        messagebox.showerror("Eroare", "Introduceti numere intre 0 si 100!")
                else:
                    messagebox.showerror("Eroare", "Introduceti toate valorile!")

            else:
                if entered_value1 and entered_value2 and entered_value3 and entered_value4:
                    if 0.0 <= entered_value1 <= 5.0 and 0.0 <= entered_value2 <= 5.0 and 0.0 <= entered_value3 <= 5.0 and 0.0 <= entered_value4 <= 5.0:
                        (id3_prediction, RN_continuous_prediction, RN_multiclass_prediction, bayes_prediction,
                         linear_regression_prediction, random_forest_prediction, ada_boost_prediction) \
                            = predict_dataset_2(entered_value1, entered_value2, entered_value3, entered_value4)
                        self.predictions_labels(id3_prediction, RN_continuous_prediction, RN_multiclass_prediction,
                                                bayes_prediction, linear_regression_prediction,
                                                random_forest_prediction, ada_boost_prediction)
                    else:
                        messagebox.showerror("Eroare", "Introduceti numere intre 0 si 5!")
                else:
                    messagebox.showerror("Eroare", "Introduceti toate valorile!")

        except ValueError:
            messagebox.showerror("Eroare", "Introduceti numere valide!")

    def on_button_data1(self):
        if self.id3_label is not None:
            self.label_algs_text.destroy()
            self.id3_label.destroy()
            self.rn_continuous_label.destroy()
            self.rn_multiclass_label.destroy()
            self.bayes_label.destroy()
            self.linear_regression_label.destroy()
            self.rf_label.destroy()
            self.ada_boost_label.destroy()

        self.css_viewer.hide_csv()
        self.css_viewer.setup_csv_section(self.path1)
        self.css_viewer.setup_csv_preprocessing_section(self.path_preprocessing1)
        self.canvas.itemconfig(self.text_label_data, text="HH_Provider_Jan2024.csv")
        self.canvas.itemconfig(self.text_label_preproc, text="Preprocessing_HH.csv")
        self.diagrams = "diagrams"
        self.canvas.itemconfig(self.text_label_col1,
                               text="- How often patients got better \n  at walking or moving around (%):")
        self.canvas.itemconfig(self.text_label_col2,
                               text="- How often patients got better \n  at getting in and out of bed (%):")
        self.canvas.itemconfig(self.text_label_col3,
                               text="- How often patients got better \n  at bathing (%):")
        self.canvas.itemconfig(self.text_label_col4, text="- How often patients' breathing \n  improved (%):")
        self.canvas.itemconfig(self.text_label_col5,
                               text="- How often patients got better \n  at taking their drugs correctly by \n  mouth (%):")
        self.entry_widget = tk.Entry(self.canvas, textvariable=self.entry_var5, width=6, font=("Arial", 13),
                                     validate="key",
                                     validatecommand=(self.canvas.register(self.validate_entry), "%P"))
        self.entry_widget.place(x=1050, y=305)

    def on_button_data2(self):
        if self.id3_label is not None:
            self.label_algs_text.destroy()
            self.id3_label.destroy()
            self.rn_continuous_label.destroy()
            self.rn_multiclass_label.destroy()
            self.bayes_label.destroy()
            self.linear_regression_label.destroy()
            self.rf_label.destroy()
            self.ada_boost_label.destroy()

        self.css_viewer.hide_csv()
        self.css_viewer.setup_csv_section(self.path2)
        self.css_viewer.setup_csv_preprocessing_section(self.path_preprocessing2)
        self.canvas.itemconfig(self.text_label_data, text="HHCAHPS_Provider_Jan2024.csv")
        self.canvas.itemconfig(self.text_label_preproc, text="Preprocessing_HHCAHPS.csv")
        self.diagrams = "diagrams2"
        self.canvas.itemconfig(self.text_label_col1,
                               text="- Star Rating for health team gave\n  care in a professional way (1-5):")
        self.canvas.itemconfig(self.text_label_col2,
                               text="- Star Rating for health team \n  communicated well with them (1-5):")
        self.canvas.itemconfig(self.text_label_col3,
                               text="- Star Rating team discussed medicines,\n  pain, and home safety (1-5):")
        self.canvas.itemconfig(self.text_label_col4,
                               text="- Star Rating for how patients rated \n  overall care from agency (1-5):")
        self.canvas.itemconfig(self.text_label_col5, text="")
        self.entry_widget.destroy()

    def on_button_stats(self):
        max_width, max_height = 0, 0

        new_window = tk.Toplevel()
        new_window.title("Statistici algoritmi de clasificare")

        images = [f"../{self.diagrams}/accuracy_comparison.png", f"../{self.diagrams}/roc_neural.png",
                  f"../{self.diagrams}/accuracy_test.png", f"../{self.diagrams}/mean_error.png"]

        # Set the desired smaller size
        new_size = (500, 350)  # Adjust the dimensions according to your preference

        for i, image_file in enumerate(images):
            file_path = os.path.join("images", image_file)
            image = Image.open(file_path)

            # Resize the image
            image = image.resize(new_size, Image.LANCZOS)

            max_width = max(max_width, image.width)
            max_height = max(max_height, image.height)

            photo = ImageTk.PhotoImage(image)

            label = tk.Label(new_window, image=photo)
            label.image = photo
            row = i // 2
            col = i % 2
            label.grid(row=row, column=col, padx=10, pady=10)

        new_window.resizable(False, False)

    def on_button_average_medians(self):
        file_path = os.path.join(self.diagrams, "medii_mediane.png")

        if file_path:
            self.show_image_in_new_window(file_path)

    def on_button_nr_inst_classes(self):
        file_path = os.path.join(self.diagrams, "nr_inst_classes.png")

        if file_path:
            self.show_image_in_new_window(file_path)

    def show_image_in_new_window(self, file_path):
        new_window = tk.Toplevel()
        new_window.title("Medii și mediane")

        image = Image.open(file_path)
        resized_image = image.resize((1000, 700), PIL.Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(resized_image)

        label = tk.Label(new_window, image=photo)
        label.image = photo
        label.pack()
        new_window.resizable(False, False)
