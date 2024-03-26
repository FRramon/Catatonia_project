# Install and load necessary packages
library(tidyverse)
library(readxl)
# Step 1: Get a list of all CSV files in the folder
csv_files <- list.files(path = "/Users/francoisramon/Desktop/These/Catatonia/data/brain_age_volbrain", pattern = "*.csv", full.names = TRUE)
print(csv_files)
datapath = "/Users/francoisramon/Desktop/These/Catatonia/data/brain_age_volbrain"
L <- substr(list.files(datapath),start = 1,7)
# Step 2: Read all CSV files and combine them into one data frame
#combined_data <- lapply(csv_files, read_csv(sep=";")) %>% bind_rows()
#subject_ids <- c("sub-1","sub-2","sub-3","sub-5","sub-7","sub-8","sub-9","sub-10","sub-11","sub-13","sub-14","sub-15","sub-17","sub-20","sub-22","sub-23","sub-24","sub-25","sub-26","sub-27","sub-29","sub-30","sub-31","sub-32","sub-33","sub-34","sub-35","sub-36")
subject_list <- gsub("_", "", L)
subject_list <- as.character(as.numeric(substr(subject_list,start=5,stop=7)))

All <- lapply(csv_files,function(i){
  read.csv(i, sep=";")
})  %>% bind_rows()


All$N_anonymisation <- subject_list
# Step 3: Write the combined data frame to a new CSV file


Statuts <- read_excel("/Users/francoisramon/Desktop/These/Catatonia/data/volume_catacof_maj.xlsx")
#Statuts$N_anonymisation #<- paste("sub-", as.character(Statuts$N_anonymisation),sep="")
Age_with_statut <- merge(All,Statuts[,c("N_anonymisation","statut","tesla","duree_pec_psy")])#,"equivalent_olz_1","equivalent_valium_10","duree_pec_psy")],by="N_anonymisation",all.x = TRUE)
write_csv(Age_with_statut, "/Users/francoisramon/Desktop/These/Catatonia/data/brain_age_vs_statut.csv")
Dtest <- read.csv("/Users/francoisramon/Desktop/These/Catatonia/data/brain_age_vs_statut.csv")

### Il manque sub 88 125 190 

age_columns <- colnames(Dtest)[grep("age", colnames(Dtest), ignore.case = TRUE)][-1]

new_cols <- paste("delta_",age_columns,sep = "")
Dtest[new_cols] <- Dtest[age_columns] - Dtest$Age

Dtest <- write.csv(Dtest,"/Users/francoisramon/Desktop/These/Catatonia/data/delta_brain_age.csv")




