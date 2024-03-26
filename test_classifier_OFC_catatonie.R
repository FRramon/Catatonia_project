source("~/Desktop/These/Catatonia/volume_mylene.R")
datapath <- "/Users/francoisramon/Desktop/These/Catatonia/volume_catacof_maj.xlsx"
DV <- read_excel(datapath)

library(readxl)
library(dplyr)
library(car)
library(glmnet)
library(pROC)
library(bestglm)

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

cols <- c("statut","Sex","tesla.x","Age","type_OFC_G","type_OFC_D",R)

#Dr <- getDataframe(D,R)
Dr <- Dall[,cols]
Dr <- Dr[!is.na(Dr$tesla.x),]
Dr[Dr == "IV"] <- "III"
Dr$type_OFC_G <- as.factor(Dr$type_OFC_G)

Dr[Dr == "IV"] <- "III"
Dr$type_OFC_D <- as.factor(Dr$type_OFC_D)


Dr$statut <- as.factor(Dr$statut)

Dr <- as.data.frame(Dr)

set.seed(1)

#use 70% of dataset as training set and 30% as test set
sample <- sample(c(TRUE, FALSE), nrow(Dr), replace=TRUE, prob=c(0.7,0.3))
train  <- Dr[sample, ]
test   <- Dr[!sample, ]


write.csv(Dr,"/Users/francoisramon/Desktop/These/Catatonia/full_catacof.csv")

SCORE <- data.frame(Y = train$statut)

# Separation en bloc

nbloc <- 5
bloc <- rep(0,nrow(train))
ind0 <- which(train$statut==0)
ind1 <- which(train$statut==1)

set.seed(1234)
bloc[ind0] <- sample(rep(1:nbloc,length=length(ind0)))
bloc[ind1] <- sample(rep(1:nbloc,length=length(ind1)))

model <- list()
for(i in 1:nbloc){
  DrA <- train[bloc!=i, ]
  DrT <- train[bloc==i, ]

  # logistique reg

  reglog <- glm(statut ~ ., data = DrA,family = "binomial")
  #S <- predict(reglog,DrT, type = "response")
  SCORE[bloc==i,"glm"] <- predict(reglog,DrT, type = "response")
  
  # mod <- lda(statut ~., data=DrA)
  # SCORE[bloc==i,"lda"] <- predict(mod,DrT,type="response")

}

Dr.X <- model.matrix(statut ~., data = train)[,-1]
Dr.Y <- train[,"statut"]


for(i in 1:nbloc){
  
  XA <- Dr.X[bloc!=i, ]
  YA <- Dr.Y[bloc!=i]
  XT <- Dr.X[bloc==i, ]
  print(paste0("Fold : ", i))
  
  # ridge
  modR <- cv.glmnet(XA,YA,alpha=0,family="binomial")

  SCORE[bloc==i,"ridge"] <- as.vector(predict(modR,XT,"lambda.1se",type = "response"))
  
  # ridge class
  modRC <- cv.glmnet(XA,YA,alpha=0,family="binomial",type.measure = "class")
  SCORE[bloc==i,"ridgeC"] <- as.vector(predict(modRC,XT,"lambda.1se",type = "response"))
  
  # ridgeAUC
  modRA <- cv.glmnet(XA,YA,alpha=0,family="binomial",type.measure = "auc")
  SCORE[bloc==i,"ridgeA"] <- as.vector(predict(modRA,XT,"lambda.1se",type = "response"))
  
  # lasso
  modL <- cv.glmnet(XA,YA,alpha=1,family="binomial")
  SCORE[bloc==i,"lasso"] <- as.vector(predict(modL,XT,"lambda.1se",type = "response"))
  
  # lasso class
  modLC <- cv.glmnet(XA,YA,alpha=1,family="binomial",type.measure = "class")
  SCORE[bloc==i,"lassoC"] <- as.vector(predict(modLC,XT,"lambda.1se",type = "response"))
  
  # lassoAUC
  modLA <- cv.glmnet(XA,YA,alpha=1,family="binomial",type.measure = "auc")
  SCORE[bloc==i,"lassoA"] <- as.vector(predict(modLA,XT,"lambda.1se",type = "response"))
  
  # elastic
  modEN <- cv.glmnet(XA,YA,alpha=0.5,family="binomial")
  SCORE[bloc==i,"elast"] <- as.vector(predict(modEN,XT,"lambda.1se",type = "response"))
  
  # elastic class
  modENC <- cv.glmnet(XA,YA,alpha=0.5,family="binomial",type.measure = "class")
  SCORE[bloc==i,"elasticC"] <- as.vector(predict(modENC,XT,"lambda.1se",type = "response"))
  
  # elasticAUC
  modENA <- cv.glmnet(XA,YA,alpha=0.5,family="binomial",type.measure = "auc")
  SCORE[bloc==i,"elasticA"] <- as.vector(predict(modENA,XT,"lambda.1se",type = "response"))
  
  # LDA
  
  
}

rocCV <- roc(Y~.,data=SCORE)
tmp <-lapply(rocCV,FUN=coords,x="best",ret=c("threshold","tp","fp","fn","tn","sensitivity","specificity","ac"),transpose=TRUE)
mat <- do.call(rbind,tmp)
aucmodele<-sort(round(unlist(lapply(rocCV,auc)),3),dec=TRUE)[1:6]
## results
print(mat)
print(aucmodele)

X =makeX(train = train,test=test)
ypred <- predict(modLC,X$xtest,"lambda.1se","response")




