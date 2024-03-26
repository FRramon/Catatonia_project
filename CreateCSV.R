source("~/Desktop/These/Catatonia/volume_mylene.R")
datapath <- "/Users/francoisramon/Desktop/These/Catatonia/volume_catacof_maj.xlsx"
DV <- read_excel(datapath)

library(readxl)
library(dplyr)
library(car)
library(glmnet)
library(pROC)
library(bestglm)
library(openxlsx)

#R <- "Cerebellum GM total volume cm3"
#R <- read.csv("/Users/francoisramon/Desktop/These/Catatonia/sign_volumes.csv")
DS <- read.csv("/Users/francoisramon/Desktop/These/Catatonia/OFC_sulcal_type_data.csv",sep=";",header = T)
## Skip sain rows
DS <- DS[DS$pathologie != "sain", ] 
DS <- DS[DS$pathologie != "Situs inversus", ] 
DS <- DS[DS$pathologie != "Situs solitus", ] 
R <- getVolumesLabels(DV)
#Th <- getThicknessLabels(DV)
DV <- as.data.frame(DV)
DS <- as.data.frame(DS)

Dall <- merge(DV,DS,by="N_anonymisation")

cols <- c("N_anonymisation","statut","ECT","Sex","tesla.x","duree_pec_psy","Age","type_OFC_G","type_OFC_D",R)

#Dr <- getDataframe(D,R)
Dr <- Dall[,cols]
#Dr <- Dr[!is.na(Dr$tesla.x),]
Dr[Dr == "IV"] <- "III"
Dr$type_OFC_G <- as.factor(Dr$type_OFC_G)

Dr[Dr == "IV"] <- "III"
Dr$type_OFC_D <- as.factor(Dr$type_OFC_D)
Dr$statut <- as.factor(Dr$statut)
Dr <- as.data.frame(Dr)

#### Add Eq valium & olz1

DV <- read_excel("/Users/francoisramon/Desktop/These/Catatonia/dataTTcatacof1.xlsx")
DV <- rename(DV, "N_anonymisation" = "anonymisation")
print(ncol(DV))

## Rename Statut


DwithV <- merge(Dr,DV,by="N_anonymisation")

#DwithV$statut <- ifelse(DwithV$statut == 0, "no catatonia", "catatonia")
DwithV <- rename(DwithV, "tesla" = "tesla.x")
DwithV$duree_pec_psy <- as.numeric(DwithV$duree_pec_psy)
DwithV$equivalent_olz_1 <- as.numeric(DwithV$equivalent_olz_1)
DwithV$equivalent_valium_10 <- as.numeric(DwithV$equivalent_valium_10)

#DwithV <- DwithV[!is.na(DwithV$equivalent_olz_1),]
#DwithV <- DwithV[!is.na(DwithV$equivalent_valium_10),]


write.csv(DwithV,"/Users/francoisramon/Desktop/These/Catatonia/full_catacof.csv")
openxlsx::write.xlsx(DwithV,"/Users/francoisramon/Desktop/These/Catatonia/full_catacof.xlsx")

datapath <- "/Users/francoisramon/Desktop/These/Catatonia/full_catacof.xlsx"
D <- read_excel(datapath)

regions <- getVolumesLabels(D)

for (r in regions){
  dHipp <- getDataframe(D,r)
}

g <- ggboxplot(D,x="statut",y="equivalent_valium_10",
               color="statut",
               palette = "jco",
               add = "jitter")
# datapath <- "/Users/francoisramon/Desktop/These/Catatonia/full_catacof.csv"
# D <- read.csv(datapath,header=T,sep=",")
# This function return a dataframe with all necessary columns for anova for a specific region.


# getDataframe <- function(D,region){
#   
#   Dr <- D[c(region,"statut","Sex","Age","ECT","tesla.x","Intracranial Cavity (IC) volume cm3", "duree_pec_psy", "equivalent_olz1" , "equivalent_valium10")]
#   
#   colnames(Dr)[1] <- "Y"
#   colnames(Dr)[7] <- "ICV"
#   
#   
#   Dr$Sex <- as.factor(Dr$Sex)
#   Dr$Age <- as.numeric(Dr$Age)
#   Dr$statut <- as.factor(Dr$statut)
#   Dr$ECT <- as.factor(Dr$ECT)
#   # Dr$tesla <- as.factor(Dr$tesla)
#   Dr$duree_pec_psy <- as.numeric(Dr$duree_pec_psy)
#   Dr$equivalent_olz1 <- as.numeric(Dr$equivalent_olz1)
#   Dr$equivalent_valium10 <- as.numeric(Dr$equivalent_valium10)
#   
#   Dr
# }
# 
# 
# Dr <- getDataframe(D,"Hippocampus total volume cm3")



