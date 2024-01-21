from logical.preprocessing import preprocessing
from logical.predictions import predictions

if __name__ == '__main__':

    preprocessing(folder_path="./diagrams", file_path="./data/HH_Provider_Jan2024.csv",
                  target_column='Quality of patient care star rating', folder_data_save="./data/Preprocessing_HH.csv",
                  folder_photo_save="./diagrams/medii_mediane.png")
    preprocessing(file_path="./data/HHCAHPS_Provider_Jan2024.csv", folder_path="./diagrams2",
                  target_column='HHCAHPS Survey Summary Star Rating',
                  folder_data_save="./data/Preprocessing_HHCAHPS.csv",
                  folder_photo_save="./diagrams2/medii_mediane.png")
    predictions(file_path="./data/Preprocessing_HH.csv", target_column='Quality of patient care star rating',
                accuracy_plot_save="./diagrams/accuracy_comparison.png", rn_cont_plot="./diagrams/accuracy_test.png",
                roc_plot="./diagrams/roc_neural.png", mae_plt="./diagrams/mean_error.png",
                folder_path_train="./trained_models",
                directory_path="./trained_models", data_view_columns="./diagrams/nr_inst_classes.png")
    predictions(file_path="./data/Preprocessing_HHCAHPS.csv", target_column='HHCAHPS Survey Summary Star Rating',
                accuracy_plot_save="./diagrams2/accuracy_comparison.png", rn_cont_plot="./diagrams2/accuracy_test.png",
                roc_plot="./diagrams2/roc_neural.png", mae_plt="./diagrams2/mean_error.png",
                folder_path_train="./trained_models2",
                directory_path="./trained_models2", data_view_columns="./diagrams2/nr_inst_classes.png")
