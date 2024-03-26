source("~/Desktop/These/Catatonia/volume_mylene.R")
datapath <- "/Users/francoisramon/Desktop/These/Catatonia/volume_catacof_maj.xlsx"
D <- read_excel(datapath)

library(readxl)
library(dplyr)
library(car)
library(glmnet)
library(pROC)
library(bestglm)

R <- getVolumesLabels(D)
#R <- "Cerebellum GM total volume cm3"
#R <- read.csv("/Users/francoisramon/Desktop/These/Catatonia/sign_volumes.csv")
#R <- R$X.Volume_name.[-1]
#R <- "Thalamus right volume cm3"
#R <- "MOrG total volume cm3"
cols <- c("statut","ECT","tesla","Age","Sex","pathologie",R)
#Dr <- getDataframe(D,R)
Dr <- D[,cols]
Dr <- Dr[!is.na(Dr$tesla),]
Dr$statut <- as.factor(Dr$statut)
Dr <- as.data.frame(Dr)



SCORE <- data.frame(Y = Dr$statut)

# Separation en bloc

nbloc <- 10
bloc <- rep(0,nrow(Dr))
ind0 <- which(Dr$statut==0)
ind1 <- which(Dr$statut==1)

set.seed(1234)
bloc[ind0] <- sample(rep(1:nbloc,length=length(ind0)))
bloc[ind1] <- sample(rep(1:nbloc,length=length(ind1)))

model <- list()
for(i in 1:nbloc){
  DrA <- Dr[bloc!=i, ]
  DrT <- Dr[bloc==i, ]

  # logistique reg

  reglog <- glm(statut ~ ., data = DrA,family = "binomial")
  #S <- predict(reglog,DrT, type = "response")
  SCORE[bloc==i,"glm"] <- predict(reglog,DrT, type = "response")
  
  # choix <- bestglm(DrA,family=binomial)
  # SCORE[bloc==i,"bestglm"] <- predict(choix$BestModel,DrT, type = "response")
  # 
  # tmp <- choix$BestModel[1,-ncol(Dr)]
  # var <- names(tmp)[tmp==TRUE]
  # models[[i]] <- var
  #print(bloc==i)
  #print(S)
}

Dr.X <- model.matrix(statut ~., data = Dr)[,-1]
Dr.Y <- Dr[,"statut"]


for(i in 1:nbloc){

  XA <- Dr.X[bloc!=i, ]
  YA <- Dr.Y[bloc!=i]
  XT <- Dr.X[bloc==i, ]
  
  
  # ridge
  mod <- cv.glmnet(XA,YA,alpha=0,family="binomial")
  print(paste0("Fold : ", i))
  SCORE[bloc==i,"ridge"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # ridge class
  mod <- cv.glmnet(XA,YA,alpha=0,family="binomial",type.measure = "class")
  SCORE[bloc==i,"ridgeC"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # ridgeAUC
  mod <- cv.glmnet(XA,YA,alpha=0,family="binomial",type.measure = "auc")
  SCORE[bloc==i,"ridgeA"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # lasso
  mod <- cv.glmnet(XA,YA,alpha=1,family="binomial")
  SCORE[bloc==i,"lasso"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # lasso class
  mod <- cv.glmnet(XA,YA,alpha=1,family="binomial",type.measure = "class")
  SCORE[bloc==i,"lassoC"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # lassoAUC
  mod <- cv.glmnet(XA,YA,alpha=1,family="binomial",type.measure = "auc")
  SCORE[bloc==i,"lassoA"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # elastic
  mod <- cv.glmnet(XA,YA,alpha=0.5,family="binomial")
  SCORE[bloc==i,"elast"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # elastic class
  mod <- cv.glmnet(XA,YA,alpha=0.5,family="binomial",type.measure = "class")
  SCORE[bloc==i,"elasticC"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
  # elasticAUC
  mod <- cv.glmnet(XA,YA,alpha=0.5,family="binomial",type.measure = "auc")
  SCORE[bloc==i,"elasticA"] <- as.vector(predict(mod,XT,"lambda.1se",type = "response"))
  
}

rocCV <- roc(Y~.,data=SCORE)
tmp <-lapply(rocCV,FUN=coords,x="best",ret=c("threshold","tp","fp","fn","tn","sensitivity","specificity","ac"),transpose=TRUE)
mat <- do.call(rbind,tmp)
aucmodele<-sort(round(unlist(lapply(rocCV,auc)),3),dec=TRUE)[1:6]
## results
print(mat)
print(aucmodele)
plot(rocCV$ridgeC,main="ROC curve - RidgeC classifier")
