import numpy as np
import pandas as pd
import scipy.stats as stats
from mulm.residualizer import Residualizer
import seaborn as sns
np.random.seed(1)


data = pd.read_excel("/Users/francoisramon/Downloads/classif.xlsx")
listcols = data.columns

cols_to_keep = [c for c in listcols if not "thickness" in c and not "asymmetry" in c and not "%" in c and not "SNR" in c and not "Scale Factor" in c and not "N_anomymisation" in c and not "tesla" in c ]
print(cols_to_keep)
data = data[cols_to_keep]
data.to_csv("/Users/francoisramon/Desktop/These/Catatonia/volumes_df_15_03.csv")
print(cols_to_keep)

res_spl = Residualizer(data, formula_res="Sex + Age + duree_TT",formula_full = "Sex + Age + duree_TT + statut  ")

print(cols_to_keep)
cols_vols = [c for c in cols_to_keep if "cm3" in c]
for col in cols_vols:
    y = data[col].to_numpy()
    print(col)
    data[col] = res_spl.fit_transform(y[:,None], res_spl.get_design_mat(data))
data.to_csv("/Users/francoisramon/Desktop/These/Catatonia/residualized_X.csv")


#print(np.mean(data[]))