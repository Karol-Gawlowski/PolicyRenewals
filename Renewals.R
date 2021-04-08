# try SMOTE - “SMOTE: Synthetic Minority Over-sampling Technique“

# ///////////////////////////////////////
# Initialization ----
# ///////////////////////////////////////

library(dplyr)
library(ggplot2)
library(caret)
library(pROC)
library(e1071)
library(glmnet)
# library(reshape2)

options(digits=2)
set.seed(100)

# Upload data
# Source: https://www.kaggle.com/arashnic/imbalanced-data-practice?select=aug_train.csv
# the testing dataset provided on the link mentioned above did not contain the actual values 
# "Response" is the target variable indicating whether a customer was successfully acquired
# "Policy_Sales_Channel" - insurance broker
data_initial=as.data.frame(read.csv("aug_train.csv"))
data=data_initial %>% arrange(id)
data$Policy_Sales_Channel=as.factor(data$Policy_Sales_Channel)
data$Region_Code=as.factor(data$Region_Code)

# we drop this variable, because it has close to 0 variance
Vintages=data %>% select(Vintage,Response)  %>%  group_by(Vintage) %>%
  mutate(Count=n_distinct(Vintage))  %>% 
  summarise(Renewals=mean(Response), Count=sum(Count)) %>% as.data.frame()
rm(Vintages)
data=data %>% select(-Vintage)

# transform factors to numeric
# Vehicle_Age: I assume that on average <1 Year == 0.5; 1-2 Year == 1.5; >2 Year == 2.5
# Vehicle_Damage: 1 for Yes, 0 for No
# Gender: 1 for Female, 0 for Male
data=data %>% mutate(Vehicle_Age=(Vehicle_Age=="< 1 Year")*0.5+
                                   (Vehicle_Age=="1-2 Year")*1.5+
                                   (Vehicle_Age==">2 Years")*2.5,
                                 Vehicle_Damage=(Vehicle_Damage=="Yes")*1,
                                 Gender=(Gender=="Female")*1)

# ///////////////////////////////////////
# data cleaning and one hot encoding ----
# ///////////////////////////////////////

# we have to clean the policy_sales_channel. 
# There is too many of them, they differ on renewal rate and volume
Sales_channels=data %>% select(Policy_Sales_Channel,Response)  %>%  group_by(Policy_Sales_Channel) %>%
                        mutate(Count=n_distinct(Policy_Sales_Channel))  %>% 
                        summarise(Renewals=mean(Response), Count=sum(Count)) %>% as.data.frame()

# we don't know anything specific about the sales channels which have low sales volume
Sales_channels=Sales_channels[order(-Sales_channels$Count,-Sales_channels$Renewals),]

# Based on the density plot below and the above, we propose 10 groups
# 1 group for the policy sales channels <=75 sales volume
# further 9 grpups based on the Renewal rate: 0-0.05; 0.05-0.1 and so on up to 0.45 
ggplot(Sales_channels %>% filter(Count>75),aes(x=Renewals))+geom_density()

# The classification is output as one-hot encoding
data$Policy_Sales_Channel=data$Policy_Sales_Channel %>% as.matrix() %>% as.integer()

Sales_channels_one_hot=matrix(ncol=10,nrow=nrow(data))


# first group 
group_1=Sales_channels %>% filter(Count<=75) %>% select(Policy_Sales_Channel) %>% as.matrix() %>% as.integer()
Sales_channels_one_hot[,1]=c(apply(data$Policy_Sales_Channel == matrix(data=rep(x=group_1,length(data$Policy_Sales_Channel)),
                                             nrow=length(data$Policy_Sales_Channel),
                                             ncol=length(group_1)),1,sum) %>% as.data.frame())[[1]]

# the other groups based on renewal/sales success rate
lower=0
sequen=seq(from=0,to=0.45,by=0.05)
colnames(Sales_channels_one_hot)=paste(rep("Sales_perc",10),sapply(sequen, toString),sep="_")

# apply one hot encoding based on % of sales in channel
for (i in 2:10){
  class_Sales_ch=Sales_channels %>% filter(Count>75) %>% filter(Renewals<=sequen[i] & Renewals>sequen[i-1]) %>% 
    select(Policy_Sales_Channel) %>% 
    as.matrix() %>% 
    as.integer() %>% sort()
  
  # return a binary vector indicating whether a given position belongs to i'th group 
  Sales_channels_one_hot[,i]=(apply(data$Policy_Sales_Channel == matrix(data=rep(x=class_Sales_ch,length(data$Policy_Sales_Channel)),
                                               nrow=length(data$Policy_Sales_Channel),
                                               ncol=length(class_Sales_ch),
                                               byrow=TRUE),1,sum) %>% as.data.frame())[[1]]
}
# consistency check - passed
sum(Sales_channels_one_hot)==nrow(data)


# substitute the one-hot encoded sales channels 
data=data %>% select(-Policy_Sales_Channel)
data=cbind(data,Sales_channels_one_hot)
rm(Sales_channels_one_hot)

# Region variable one hot encoding 
Regions=predict(dummyVars("~.",data=data$Region_Code %>% as.data.frame),
        newdata=data$Region_Code %>% as.data.frame)
colnames(Regions)=paste(rep("Region",length(unique(data$Region_Code))),0:52,sep="_")

data=data %>% select(-Region_Code)
data=cbind(data,Regions)

# check passed
unique(rowSums(Regions))==1
rm(Regions)

# Annual premium
# there are instances with very high premium but less than 10% observ are above k=50,000 
# and there is over 4% point difference between the mean response in these two groups

k=50000
quantile(data$Annual_Premium ,probs = seq(0,1,by=0.05))

data %>% select(Annual_Premium,Response) %>% 
         mutate(Indicator=(Annual_Premium>k)*1) %>% 
         group_by(Indicator) %>% 
         summarise(mean=mean(Response))

# Create an additional column of values in excess of k
Premiums=data %>% select(Annual_Premium) %>% mutate(Indicator=(Annual_Premium>k)*1,
                                                    Premium_Surplus=Indicator*(Annual_Premium-k),
                                                    Annual_Premium=Annual_Premium-Premium_Surplus) %>% 
                                             select(-Indicator)

Premiums$Premium_Surplus=Normalize(Premiums$Premium_Surplus)
Premiums$Annual_Premium=Normalize(Premiums$Annual_Premium)

data=data %>% select(-Annual_Premium)
data=cbind(data,Premiums)
rm(Premiums)

# Normalize age
data$Age=Normalize(data$Age)

# ///////////////////////////////////////
# feature engineering  ----
# ///////////////////////////////////////

# think about creating a new variable from region x sales channel. 

data_PCA=prcomp(x = data)


# ///////////////////////////////////////
# train/test split  ----
# ///////////////////////////////////////

sample=c(sample_frac(as.data.frame(1:nrow(data)),size = 0.8))[[1]]
train_data=data[sample,]
test_data=data[-sample,]


# ///////////////////////////////////////
# resampling to balance ----
# ///////////////////////////////////////

# initial imbalance in the training set 16%
mean(train_data$Response)

# reducing the imbalance to 28% by resampling, on average is equivalent to just 
# copying the instances that renewed the policy
train_data_resampled=rbind(train_data,train_data %>% filter(Response==1))
mean(train_data_resampled$Response)

# ///////////////////////////////////////
# EDA ----
# ///////////////////////////////////////

# Correlations
correlations=cor(train_data[,-c(1,5,10)])%>% as.data.frame()
correlations*(correlations>=0.1)+correlations*(correlations<=-0.1)

# check conditional plots/statistics given someone actually renewed the policy or not

summary(train_data[,-1] %>% filter(Response==1))
summary(train_data[,-1] %>% filter(Response==0))

Renewal_yes=train_data[,-1] %>% filter(Response==1)
Renewal_no=train_data[,-1] %>% filter(Response==0)

# Plots
# 1.
# split of renewals by age suggests  that for ages approx 20-30 people don't renew (peak around 23)
# but for ages 30-60 they do (peak around 45)
#further splits by age conditional on gender or on premium found not to be significant
ggplot(train_data ,aes(x=Age))+
                                geom_density(aes(colour=factor(Response)),size=1)+
                                ggtitle("Renewals by Age")

ggplot(train_data_resampled ,aes(x=Age))+
                                geom_density(aes(colour=factor(Response)),size=1)+
                                ggtitle("Renewals by Age (resampled)")

# 2.
# renewals by region
# there are regions with renewal rate above 20% and up to ~27.5%
train_data_for_histogram=data_initial %>% select(Response,Region_Code) %>% group_by(Region_Code) %>% summarise(Renewals=mean(Response)) %>% as.data.frame()
train_data_for_histogram=train_data_for_histogram[order(train_data_for_histogram$Renewals),]

ggplot(train_data_for_histogram, aes(x=reorder(Region_Code,Renewals),y=Renewals))+
                                geom_bar(stat="identity")+
                                ggtitle("Renewals by region")

# 3.
# renewals by region and gender (mostly the same distribution)
ggplot(data_initial %>% select(Response,Region_Code,Gender) %>% group_by(Region_Code,Gender) %>% 
                          summarise(Renewals=mean(Response)) %>% as.data.frame(), 
                                aes(x=reorder(Region_Code,Renewals),y=Renewals,fill=factor(Gender)))+
                                geom_bar(stat="identity")+
                                ggtitle("Renewals by region",subtitle = "Split by Gender")+
                                xlab('Region code')+
                                ylab("Renewals proportion")

# 4.
ggplot(data_initial %>% select(Policy_Sales_Channel,Response)  %>% 
                        filter(Policy_Sales_Channel %ni% group_1) %>%  
                        group_by(Policy_Sales_Channel) %>% 
                        mutate(Count=n_distinct(Policy_Sales_Channel))  %>% 
                        summarise(Renewals=mean(Response), Count=sum(Count)) %>% 
                        filter(Renewals>0) %>% 
                        as.data.frame(), 
                          aes(x=reorder(Policy_Sales_Channel,Renewals),y=Renewals))+
                                geom_bar(stat="identity")+
                                ggtitle("Renewals by region")+
                                xlab('Region code')+
                                ylab("Renewals proportion")

# 5.
# those who renew are offered basically the same premium 
ggplot(data_initial ,aes(x=Annual_Premium))+
                                geom_density(aes(colour=factor(Response)),size=1)+
                                ggtitle("Renewals by annual premium") +
                                xlim(0,100000)

# ///////////////////////////////////////
# Modeling - setting up ---- <- NOT RUN, STILL IN DEVELOPMENT
# ///////////////////////////////////////
# parallel processing
library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)


trControl = trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 2)

# ## When you are done:
# stopCluster(cl)

# ///////////////////////////////////////
# Modeling - LOGISTIC REGRESSION ----
# ///////////////////////////////////////

# trControl=trainControl(method = "CV",number = 10)

# on regular data 
# Logistic_reg=glm(train_data[,-1],formula = Response ~.,family = "binomial")

Logistic_reg=train(x= train_data %>% select(-id,-Response) , y=as.factor(train_data$Response),
                                                                   method = 'glmnet',
                                                                   # trControl = trControl,
                                                                   family = 'binomial' )

treshold=0.4
Predict_Logistic_reg=predict(Logistic_reg,
                             test_data %>% select(-id,-Response),
                             type = "prob")

Predict_Logistic_reg_fact=as.factor((predict(Logistic_reg,
                                             test_data %>% select(-id,-Response),
                                             type = "prob")[,1] > treshold)*1)

confusionMatrix(data = Predict_Logistic_reg_fact,
                reference = as.factor(test_data$Response))

roc(predictor = predict(Logistic_reg,
             test_data %>% select(-id,-Response),
             type = "prob")[,1],
    response = as.factor(test_data$Response),plot = TRUE)




  
# on RESAMPLED data  
Logistic_reg_resampled=train(x= train_data %>% select(-id,-Response) , y=as.factor(train_data$Response),
                   method = 'glmnet',
                   # trControl = trControl,
                   family = 'binomial' )

treshold=0.4
Predict_Logistic_reg_resampled=predict(Logistic_reg_resampled,
                             test_data %>% select(-id,-Response),
                             type = "prob")

Predict_Logistic_reg_fact_resampled=as.factor((predict(Logistic_reg_resampled,
                                             test_data %>% select(-id,-Response),
                                             type = "prob")[,1] > treshold)*1)

confusionMatrix(data = Predict_Logistic_reg_fact_resampled,
                reference = as.factor(test_data$Response))

roc(predictor = predict(Logistic_reg_resampled,
                        test_data %>% select(-id,-Response),
                        type = "prob")[,1],
    response = as.factor(test_data$Response),plot = TRUE)



# ///////////////////////////////////////
# Modeling - Neural Network ----
# ///////////////////////////////////////
NeuralNet = train(Response ~ ., data = train_data %>% select(-id) %>% mutate(Response=as.factor(Response)), 
                  trControl = trControl,
                  method = "mlp",
                  size=100)

treshold_nn=0.5
Predict_NeuralNet=predict(NeuralNet,
                          test_data %>% select(-id,-Response),
                          type = "prob")

Predict_NeuralNet_reg_fact=as.factor((predict(NeuralNet,
                                      test_data %>% select(-id,-Response),
                                      type = "prob")[,2] > treshold_nn)*1)

confusionMatrix(data = Predict_NeuralNet_reg_fact,
                reference = as.factor(test_data$Response))

roc(predictor = predict(NeuralNet,
                        test_data %>% select(-id,-Response),
                        type = "prob")[,2],
    response = as.factor(test_data$Response),plot = TRUE)
 
# on resampled data
NeuralNet_Resampled = train(Response ~ ., data = train_data_resampled %>% select(-id) %>% mutate(Response=as.factor(Response)), 
                  trControl = trControl,
                  method = "mlp",
                  size=100)

treshold_nn_Resampled=0.6
Predict_NeuralNet_Resampled=predict(NeuralNet_Resampled,
                          test_data %>% select(-id,-Response),
                          type = "prob")

Predict_NeuralNet_reg_fact_Resampled=as.factor((predict(NeuralNet_Resampled,
                                              test_data %>% select(-id,-Response),
                                              type = "prob")[,2] > treshold_nn_Resampled)*1)

confusionMatrix(data = Predict_NeuralNet_reg_fact_Resampled,
                reference = as.factor(test_data$Response))

NeuralNet_Resampled_ROC=roc(predictor = predict(NeuralNet_Resampled,
                        test_data %>% select(-id,-Response),
                        type = "prob")[,2],
    response = as.factor(test_data$Response),
    plot = TRUE,
    ret=TRUE)
  
# sth aint workin
# Random Forest
RandomForest = train(Response ~ ., data = train_data %>% select(-id) %>% mutate(Response=as.factor(Response)), 
                            trControl = trControl,
                            method = "ranger")

treshold_RF=0.5
Predict_RandomForest=predict(RandomForest,
                                    test_data %>% select(-id,-Response),
                                    type = "prob")

# STH AINT WORKING
RandomForest_reg_fact=as.factor((predict(RandomForest,
                                                        test_data %>% select(-id,-Response),
                                                        type = "raw")[,2] > treshold_RF)*1)

confusionMatrix(data = RandomForest_reg_fact,
                reference = as.factor(test_data$Response))

Random_Forest_ROC=roc(predictor = predict(RandomForest_reg_fact,
                                                test_data %>% select(-id,-Response),
                                                type = "prob")[,2],
                            response = as.factor(test_data$Response),
                            plot = TRUE,
                            ret=TRUE)






# ///////////////////////////////////////
# notes ----
# ///////////////////////////////////////




# check if the dependency eg on age is linear and whether there is a significant difference among two groups
# if yes - then add +/-1 to the age (thus create two extra data instances that bought the policy)




# Pre-processing - PCA, LDA/QDA + Logistic Regression, NN, RF

# Neural network -> logistic regression (like suggested by Mario Wutrich)



# Explainability

