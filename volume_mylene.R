library(readxl)
library(car)
library(ggpubr)
library(dplyr)


#### Define utils functions for whole statistics rmarkdown script ####

# This function returns the column names list corresponding to regional volumes in cm3
getVolumesLabels <- function(D){
  C <- D %>%
    dplyr::select(contains("cm3"))
  colnames(C)
}

# This function returns the column names list corresponding to cortical thickness in mm
getThicknessLabels <- function(D){
  C <- D %>%
    dplyr::select(contains("thickness mm"))
  colnames(C)
}

getAgeLabels <- function(D){
  C <- D %>%
    dplyr::select(contains("age"))
  colnames(C)
}

# This function return a dataframe with all necessary columns for anova for a specific region.
getDataframe <- function(D,region){

  Dr <- D[c(region,"statut","Sex","Age","ECT","tesla","Intracranial Cavity (IC) volume cm3", "duree_pec_psy" , "equivalent_olz_1" , "equivalent_valium_10")]
  
  colnames(Dr)[1] <- "Y"
  colnames(Dr)[7] <- "ICV"
  
  
  Dr$Sex <- as.factor(Dr$Sex)
  Dr$Age <- as.numeric(Dr$Age)
  Dr$statut <- as.numeric(Dr$statut)
  Dr$statut <- ifelse(Dr$statut == 0, "No catatonia", "Catatonia")
  Dr$statut <- as.factor(Dr$statut)
  Dr$ECT <- as.factor(Dr$ECT)
 # Dr$tesla <- as.factor(Dr$tesla)
  Dr$duree_pec_psy <- as.numeric(Dr$duree_pec_psy)
  Dr$equivalent_olz_1 <- as.numeric(Dr$equivalent_olz_1)
  Dr$equivalent_valium_10 <- as.numeric(Dr$equivalent_valium_10)
  
  Dr
}

getDataframeAge <- function(D,region){
  
  Dr <- D[c(region,"statut","Sex","Age","tesla","duree_pec_psy")]# , "equivalent_olz_1" , "equivalent_valium_10")]
  
  colnames(Dr)[1] <- "Y"
 # colnames(Dr)[7] <- "ICV"
  Dr$Age <- as.numeric(Dr$Age)
  
  Dr$Y <- Dr$Y
  
  
  Dr$Sex <- as.factor(Dr$Sex)
  Dr$statut <- as.numeric(Dr$statut)
  Dr$statut <- ifelse(Dr$statut == 0, "No catatonia", "Catatonia")
  Dr$statut <- as.factor(Dr$statut)
  
 # Dr$ECT <- as.factor(Dr$ECT)
  Dr$tesla <- as.factor(Dr$tesla)
  Dr$duree_pec_psy <- as.numeric(Dr$duree_pec_psy)
  # Dr$equivalent_olz_1 <- as.numeric(Dr$equivalent_olz_1)
  # Dr$equivalent_valium_10 <- as.numeric(Dr$equivalent_valium_10)
  # 
  Dr
}

isSignificant <- function(res_df){
  if(res_df$`Pr(>F)`[1]<0.05){
    return("Significant effect of statut")
  }else{
    return("")
  }
}

isSignificant2 <- function(res_df){
  D <- 0
  if(res_df$`Pr(>F)`[1]<0.05){
    D <- 1
  }
  D
}

##### Stats ####

# g <- ggboxplot(Dr,x="statut",y="Y",
#                        color="statut",
#                        palette = "jco",
#                        add = "jitter")
# g


## Table 1

# D <- readxl::read_excel("/Users/francoisramon/Desktop/These/Catatonia/classif.xlsx")
# D$equivalent_olz_1 <- as.numeric(D$equivalent_olz_1)
# D$equivalent_valium_10 <- as.numeric(D$equivalent_valium_10)
# 
# D1 <- D[D$statut==1,]
# D0 <- D[D$statut==0,]
# 
# t.test(D1$equivalent_valium_10,D0$equivalent_valium_10)
# t.test(D1$equivalent_olz_1,D0$equivalent_olz_1)
# 
# t.test(D$equivalent_olz_1~D$statut)
# t.test(D$equivalent_valium_10~D$statut)
# 
# mean(D1$equivalent_valium_10,na.rm=TRUE)
# mean(D0$equivalent_valium_10,na.rm=TRUE)
# 
# sd(D1$equivalent_valium_10,na.rm=TRUE)
# sd(D0$equivalent_valium_10,na.rm=TRUE)
# 
# mean(D1$equivalent_olz_1,na.rm=TRUE)
# mean(D0$equivalent_olz_1,na.rm=TRUE)
# 
# sd(D1$equivalent_olz_1,na.rm=TRUE)
# sd(D0$equivalent_olz_1,na.rm=TRUE)


### FDR correction 

# results_cov <- read.csv("/Users/francoisramon/Desktop/These/Catatonia/results/results_normalized.csv",header=T)
# colnames(results_cov) <- c("id","volname","pvalue")
# p <- results_cov$pvalue
# results_cov$padj <- p.adjust(p, method = "fdr")
# Rsign <- results_cov[results_cov$padj<0.05,]
# sorted <- sort(results_cov$padj)
# sorted
