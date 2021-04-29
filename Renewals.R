# try SMOTE - “SMOTE: Synthetic Minority Over-sampling Technique“

# ///////////////////////////////////////
# Initialization ----
# ///////////////////////////////////////
install_version("DMwR", "0.4.1") #Package ‘DMwR’ was removed from the CRAN repository.
library(DMwR)
library(tidyverse)
library(tidyquant)
library(caret)
library(pROC)
library(e1071)
library(glmnet)
library(remotes)
library(reshape2)

options(digits=2)
set.seed(100)

# Upload data
# Source: https://www.kaggle.com/arashnic/imbalanced-data-practice?select=aug_train.csv
# the testing dataset provided on the link mentioned above did not contain the actual values 
# "Response" is the target variable indicating whether a customer was successfully acquired
# "Policy_Sales_Channel" - insurance broker
data_initial=as.data.frame(read.csv("aug_train.csv")) %>% as_tibble()
data=data_initial %>% arrange(id)

# there are no missing data
sum(is.na(data))==1
print(data)

# change data type to factors where necessary and apply binary encoding
data=data %>% mutate(Policy_Sales_Channel=as.factor(Policy_Sales_Channel),
                     Region_Code=as.factor(Region_Code),
                     Driving_License=as.factor(Driving_License),
                     Response=as.factor(Response),
                     Vehicle_Age=as.factor(Vehicle_Age),
                     Vehicle_Damage=as.factor((Vehicle_Damage=="Yes")*1),
                     Gender=as.factor((Gender=="Female")*1))

# one hot encoding of vehicle_age
OneHot = dummyVars("~ Vehicle_Age", data=data) %>% 
         predict(newdata = data) %>% 
         as.data.frame() %>% 
         rename(Car_Age_New=1,
                Car_Age_Oldest=2,
                Car_Age_Older=3)

data=data %>% select(-Vehicle_Age) %>% cbind(OneHot)
rm(OneHot)

# ///////////////////////////////////////
# Initial EDA  ----
# ///////////////////////////////////////
# Distribution of the response variable by gender
data %>% 
  ggplot(aes(x=Response))+
  geom_bar(aes(y = (..count..)/sum(..count..)))+
  scale_y_continuous(labels=scales::percent) +
  facet_grid(~Gender)+
  ylab("") +
  ggtitle("Distribution of the response variable",
          subtitle = "It's an imbalanced dataset")

# Regions with the highest renewal rates by gender
data %>% 
  mutate(Response=as.integer(Response)-1) %>% 
  select(Region_Code,Response,Gender) %>% 
  pivot_table(.rows = Region_Code,
              .columns = Gender,
              .values = ~mean(Response)) %>% 
  rename(Male=2,
         Female=3) %>% 
  arrange(-Female,-Male)

# Regions with the highest mean premium 
data %>% 
  # mutate(Response=as.integer(Response)-1) %>% 
  select(Region_Code,Annual_Premium,Gender) %>% 
  pivot_table(.rows = Region_Code,
              .columns = Gender,
              .values = ~mean(Annual_Premium)) %>% 
  rename(Male=2,
         Female=3) %>% 
  arrange(-Female,-Male)



# we note that there are policyholders with very high premiums,
# for the needs of the preliminary EDA, we will focus on the 95% of observations 
# below 55 200 annual premium
quantile(data$Annual_Premium,probs=seq(0,1,by=0.05))
# and it appears that 16% of the insureds pay 2630 
sum(data$Annual_Premium==2630)/nrow(data)

# to do: a quick look at the outliers and comparison between the majority  
data %>%
  filter(Annual_Premium>=55200) %>% 
  summary() 
  
data %>%
  filter(Annual_Premium>=55200) %>% 
  apply(2,mean)

# summary of mean premium by gender and age
# set the proportion of data to keep outside confidence bands
Q=0.1
Prem_by_gen_age=cbind(
                      data %>%
                      filter(Annual_Premium<=55200) %>% 
                      pivot_table(.rows = Age,
                                  .columns = Gender,
                                  .values = ~mean(Annual_Premium)) %>% 
                       rename(Male=2,
                              Female=3), 

                      # lower band
                      data %>%
                      filter(Annual_Premium<=55200) %>% 
                      pivot_table(.rows = Age,
                                  .columns = Gender,
                                  .values = ~quantile(Annual_Premium,probs=Q/2)) %>% 
                      rename(Male_lower=2,
                             Female_lower=3) %>% select(-Age), 

                      # upper band
                      data %>%
                      filter(Annual_Premium<=55200) %>% 
                      pivot_table(.rows = Age,
                                  .columns = Gender,
                                  .values = ~quantile(Annual_Premium,probs=1-Q/2)) %>% 
                      rename(Male_upper=2,
                             Female_upper=3) %>% select(-Age) 
)

# work in progress (to add confidence bands )
Prem_by_gen_age[1:1000,] %>% 
  ggplot(aes(x=Age,y=Annual_Premium))+
  geom_point(aes(color=Gender))+
  ggtitle("Distribution of mean premium by age")

# mean premium profile by gendar and age
Prem_by_gen_age[,1:3] %>% 
  gather(key = "Gender",
         value="Premium",
         -Age) %>% 
  ggplot(aes(x=Age,y=Premium,Fill=Gender))+
  geom_point()+
  ggtitle("Distribution of mean premium by age")

# excel style pivot table
data %>%
  pivot_table(.rows = Age,
              .columns = Gender,
              .values = ~mean(Annual_Premium))

# vintage variable - we need to split it, as the majority has small variance,
# but it might be interesting to separate its tails
data %>% 
  ggplot(aes(x=Vintage))+
  geom_density()+
  ggtitle("Density of Vintage variable")+
  xlab("")+
  ylab("")

# clean the env
rm(Prem_by_gen_age,Q)

# ///////////////////////////////////////
# data cleaning  ----
# ///////////////////////////////////////

# like above - we have to separate out the tails of this variable
# Vintages=data %>% 
#   select(Vintage,Response)  %>%  
#   group_by(Vintage) %>%
#   mutate(Count=n_distinct(Vintage))  %>% 
#   summarise(Renewals=mean(Response), Count=sum(Count)) %>% 
#   as.data.frame()
# 
# rm(Vintages)
# data=data %>% select(-Vintage)

# we have to clean the policy_sales_channel. 
# There is too many of these channels, they differ on renewal rate and volume
# we don't know anything specific about the sales channels which have low sales volume
Sales_channels=data %>% 
               mutate(Response=as.integer(Response)-1) %>% 
               select(Policy_Sales_Channel,Response)  %>%  
               group_by(Policy_Sales_Channel) %>%
               mutate(Count=n_distinct(Policy_Sales_Channel))  %>% 
               summarise(Renewals=mean(Response), Count=sum(Count)) %>% 
               arrange(-Count,-Renewals) %>% 
               as.data.frame()

# Based on the density plot below and the above observations, we propose 10 groups
# 1 group for the policy sales channels <=75 sales volume
# further 9 grpups based on the Renewal rate: 0-0.05; 0.05-0.1 and so on up to 0.45 
Sales_channels %>% 
                 filter(Count>75) %>% 
                 ggplot(aes(x=Renewals))+
                 geom_density()+
                 ggtitle("Renewals distribution, by sales channel")+
                 xlab("")+
                 ylab("")
  
# The classification is output as one-hot encoding 
data$Policy_Sales_Channel=data$Policy_Sales_Channel %>% as.matrix() %>% as.integer()
Sales_channels_one_hot=matrix(ncol=10,nrow=nrow(data))

# first group, for channels with volume of <= 75 clients
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
  class_Sales_ch=Sales_channels %>% 
                 filter(Count>75) %>% 
                 filter(Renewals<=sequen[i] & Renewals>sequen[i-1]) %>% 
                 select(Policy_Sales_Channel) %>% 
                 as.matrix() %>% 
                 as.integer() %>% 
                 sort()
  
  # return a binary vector indicating whether a given position belongs to i'th group 
  Sales_channels_one_hot[,i]=(apply(data$Policy_Sales_Channel == matrix(data=rep(x=class_Sales_ch,length(data$Policy_Sales_Channel)),
                                               nrow=length(data$Policy_Sales_Channel),
                                               ncol=length(class_Sales_ch),
                                               byrow=TRUE),1,sum) %>% as.data.frame())[[1]]
}
# consistency check
sum(Sales_channels_one_hot)==nrow(data)

# substitute the one-hot encoded sales channels 
data=data %>% select(-Policy_Sales_Channel)
data=cbind(data,Sales_channels_one_hot)

# clean env
rm(Sales_channels_one_hot,sequen,lower,group_1,class_Sales_ch,Sales_channels)

# Region variable one hot encoding 
# unluckilly, we don't know much about these regions,
# it might be a good idea to try to group them - to consider
Regions=predict(dummyVars("~.",data=data$Region_Code %>% as.data.frame),
                newdata=data$Region_Code %>% as.data.frame)

colnames(Regions)=paste(rep("Region",
                            length(unique(data$Region_Code))),
                        0:52,sep="_")

data=data %>% select(-Region_Code)
data=cbind(data,Regions)

# check 
unique(rowSums(Regions))==1
rm(Regions)

# Annual premium
# there are instances with very high premium but less than 5% observations are above k=52,000 
# we observe, that the renewal rate is 25% higher in case of high premium instances
k=52000
quantile(data$Annual_Premium ,probs = seq(0,1,by=0.05))

data %>% mutate(Response=as.integer(Response)-1) %>% 
         select(Annual_Premium,Response) %>% 
         mutate(Indicator=(Annual_Premium>k)*1) %>% 
         group_by(Indicator) %>%
         summarise(mean=mean(Response))

# Create an additional column of values in excess of k 
# works more efficiently through use of indicator, than rowwise function
Premiums=data %>% select(Annual_Premium) %>% mutate(Indicator=(Annual_Premium>k)*1,
                                                    Premium_Surplus=Indicator*(Annual_Premium-k),
                                                    Annual_Premium=Annual_Premium-Premium_Surplus) %>% 
                                             select(-Indicator)

# Normalization (remember to run the function.R script first)
Premiums$Premium_Surplus=Normalize(Premiums$Premium_Surplus)
Premiums$Annual_Premium=Normalize(Premiums$Annual_Premium)

data=data %>% select(-Annual_Premium)
data=cbind(data,Premiums)
rm(Premiums,k)

# Normalize age
data$Age=Normalize(data$Age)

# ///////////////////////////////////////
# feature engineering  ----
# ///////////////////////////////////////


# TBD

# ///////////////////////////////////////
# train/test split----
# ///////////////////////////////////////

sample=c(sample_frac(as.data.frame(1:nrow(data)),size = 0.8))[[1]]
train_data=data[sample,]
test_data=data[-sample,]

# ///////////////////////////////////////
# resampling to balance ----
# ///////////////////////////////////////

# the SMOTE function takes forever to execute. Since the package is officially archived, 
# I might redo this algorithm 

# NOT RUN
# initial imbalance in the training set 16%
# mean(train_data$Response)
# 
# train_data=rbind(train_data,
#                  SMOTE(form = Response ~ ., 
#                        data = train_data %>% select(-id) %>% mutate(Response=factor(Response)), 
#                        perc.over = 200, 
#                        perc.under = 0,
#                        k=5))
#   
# 
# nrow(a)
# 
# summary(a)
# summary(train_data[1:1000,])
# 
# mean(as.numeric(a$Response)-1)
# 
# tail(a)

# ///////////////////////////////////////
# EDA ----
# ///////////////////////////////////////

# Correlations
correlations=cor(train_data[,-c(1,5,10)]) %>% as.data.frame()
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


train(x=train_data,
      method = "mlpSGD",
      size= ,        #(Hidden Units)
      l2reg= ,       #(L2 Regularization)
      lambda= ,      #(RMSE Gradient Scaling)
      learn_rate= ,  #(Learning Rate)
      momentum= ,    #(Momentum)
      gamma= ,       #(Learning Rate Decay)
      minibatchsz= , #(Batch Size)
      repeats =      #(Models)
      )
      

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

