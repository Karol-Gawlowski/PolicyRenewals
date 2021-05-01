# try SMOTE - “SMOTE: Synthetic Minority Over-sampling Technique“

# ///////////////////////////////////////
# Initialization ----
# ///////////////////////////////////////
# install_version("DMwR", "0.4.1") #Package ‘DMwR’ was removed from the CRAN repository.
library(DMwR) # incl. SMOTE - Synthetic Minority Oversampling TEchnique - Nitesh Chawla, et al. 2002 
library(ROSE) # incl ROSE - Training and assessing classification rules with unbalanced data Menardi G.,Torelli N. 2013
library(tidyverse)
library(tidyquant)
library(caret)
library(pROC)
library(e1071)
library(glmnet)
library(remotes)
library(reshape2)
library(DALEX)

options(digits=2)
seed=100
set.seed(seed)

# Upload data
# Source: https://www.kaggle.com/arashnic/imbalanced-data-practice?select=aug_train.csv
# the testing dataset provided on the link mentioned above did not contain the actual values 
# "Response" is the target variable indicating whether a customer was successfully acquired
# "Policy_Sales_Channel" - insurance broker
data_initial=as.data.frame(read.csv("aug_train.csv")) %>% as_tibble()
data=data_initial %>% arrange(id)

# there are no missing data
sum(is.na(data))==0
print(data)

# change data type to factors where necessary and apply binary encoding
data=data %>% mutate(Policy_Sales_Channel=as.factor(Policy_Sales_Channel),
                     Region_Code=as.factor(Region_Code),
                     Driving_License=as.factor(Driving_License),
                     # Response=as.factor(Response), # it is more convenient to do that just before modeling
                     Vehicle_Age=as.factor(Vehicle_Age),
                     Vehicle_Damage=as.factor((Vehicle_Damage=="Yes")*1),
                     # Gender=as.factor((Gender=="Female")*1) # it is more convenient for plotting 
                     )

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
data %>% mutate(Response=as.factor(Response)) %>% 
         ggplot(aes(x=Response))+
         geom_bar(aes(y = (..count..)/sum(..count..)))+
         scale_y_continuous(labels=scales::percent) +
         facet_grid(~Gender)+
         ylab("") +
         ggtitle("Distribution of the response variable",
                 subtitle = "It's an imbalanced dataset")

# Regions with the highest renewal rates by gender
data %>% 
  select(Region_Code,Response,Gender) %>% 
  pivot_table(.rows = Region_Code,
              .columns = Gender,
              .values = ~mean(Response)) %>% 
  arrange(-Female,-Male)

# Regions with the highest mean premium 
data %>% 
  select(Region_Code,Annual_Premium,Gender) %>% 
  pivot_table(.rows = Region_Code,
              .columns = Gender,
              .values = ~mean(Annual_Premium)) %>% 
  arrange(-Female,-Male)

# note that there are policyholders with very high premiums,
# for the needs of the preliminary EDA, lets focus on the 95% of observations 
# below 55 200 annual premium
quantile(data$Annual_Premium,probs=seq(0,1,by=0.05))

# and it appears that 16% of the insureds pay 2630 
sum(data$Annual_Premium==2630)/nrow(data)

# Quick comparison between the high/low premium insureds
data %>% 
  mutate(High_Prem=(Annual_Premium>=55200)*1) %>% 
  group_by(High_Prem) %>% 
  summarise(n=n(),
            Renewals=mean(Response),
            Females=mean(Gender=="Female"),
            Age=mean(Age),
            D_License=mean(Driving_License==1),
            New_cars=mean(Car_Age_New),
            Older_cars=mean(Car_Age_Older),
            Oldest_cars=mean(Car_Age_Oldest))
# work in progress to plot the above table 
#           ) %>% 
  # t() %>% 
  # as.data.frame() %>% 
  # rownames_to_column() %>% 
  # rename(Variable=1,LowPrem=2,HighPrem=3) %>% 
  # gather("Lowprem","HighPrem",-Variable) %>% 
  # ggplot(aes(x=Variable))+
  # barplot(y=HighPrem)+
  # facet_grid(~LowPrem)
  
# summary of mean premium by gender and age
# set the proportion of data to keep outside the confidence bands
Q=0.2
# for 80% of observations the spread of the premiums 
# doesn't allow to infer anything interesting
data %>%
  filter(Annual_Premium<=52000 & Annual_Premium>=2630) %>% 
  select(Gender,Age,Annual_Premium) %>% 
  group_by(Age,Gender) %>% 
  summarise(
    LowerBound=quantile(Annual_Premium,probs=Q/2),
    Mean=mean(Annual_Premium),
    UpperBound=quantile(Annual_Premium,probs=1-Q/2)
  ) %>% 
  ggplot(aes(x=Age,y=Mean,color=Gender))+
  # geom_point()+
  geom_line()+
  geom_ribbon(aes(ymin=LowerBound,
                  ymax=UpperBound),
              alpha=0.3)+
  ylab("Annual Premium") +
  ggtitle("Annual Premium by Gender and Age",
          subtitle = "")

# renewals by age
# between 25-30% of people aged 30-50 renew/buy the insurance
data %>%
  group_by(Gender,Age) %>% 
  summarise(Renewals=mean(Response)) %>% 
  ggplot(aes(x=Age,y=Renewals,color=Gender))+
  geom_line()+  
  ylab("Renewal Rate") +
  ggtitle("Renewal Rate by Age and Gender")

# renewals by region
# it appears that the region with the largest client base (region 28), is also the 2nd best 
# when it comes to renewal rate
data %>%
  group_by(Region_Code) %>% 
  summarise(RegionPopulation=n(),
            Renewals=mean(Response)) %>%  # arrange(-Renewals) %>% print(n=100)
  # filter(Renewals>mean(Renewals)) %>% 
  ggplot(aes(x=reorder(Region_Code,Renewals),y=Renewals,fill=RegionPopulation))+
  geom_bar(stat="identity")+
  ylab("Renewal Rate") +
  xlab("Region Code") +
  ggtitle("Regions with most renewals")

# renewals by policy sales channel 
# getting rid of vendors, which have less than K customers
K=150
data %>%
  group_by(Policy_Sales_Channel) %>% 
  summarise(Customers=n(),
            Renewals=mean(Response)) %>%  # arrange(-Customers) %>% print(n=100)
  filter(Customers>K) %>%
  ggplot(aes(x=reorder(Policy_Sales_Channel,Renewals),y=Renewals,fill=Customers))+
  geom_bar(stat="identity")+
  ylab("Renewal Rate") +
  xlab("Broker ID") +
  ggtitle("Brokers and renewal rates")

sum((data$Vintage<=25)*1 + (data$Vintage>=285)*1)/length(data$Vintage)

# vintage variable - it should be split, as the majority has small variance,
# but it might be interesting to separate its tails
# note - we don't know the meaning of this variable + it appears as it has 
# been later deleted from kaggle 
# the horizontal dashed lines at 25 and 285 indicate the tails of the variable which
# seem to contain some information, other than the majority of the variable 
data %>% 
  ggplot(aes(x=Vintage))+
  geom_density()+
  geom_vline(xintercept=25, 
             color = "#FC4E08", 
             linetype = "dashed", 
             size = 0.5)+
  geom_vline(xintercept=285, 
             color = "#FC4E08", 
             linetype = "dashed", 
             size = 0.5)+
  xlab("")+
  ylab("")+
  ggtitle("Density of Vintage variable",
          subtitle = "dashed lines for x={25,285}")

# 11% of observations are outside of 25,285 interval
sum((data$Vintage<=25)*1 + (data$Vintage>=285)*1)/length(data$Vintage)


# excel style pivot table
data %>%
  pivot_table(.rows = Age,
              .columns = Gender,
              .values = ~mean(Annual_Premium))



# Correlations
correlations=cor(data[,-c(1,5,10)]) %>% as.data.frame()
correlations*(correlations>=0.1)+correlations*(correlations<=-0.1)

# clean the env
rm(Prem_by_gen_age,Q,K)
# ///////////////////////////////////////
# EDA ----
# ///////////////////////////////////////



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

# ///////////////////////////////////////
# data cleaning  ----
# ///////////////////////////////////////

# like mentioned above - lets separate out the tails of this variable
# Vintages=data %>% 
#   select(Vintage,Response)  %>%  
#   group_by(Vintage) %>%
#   mutate(Count=n_distinct(Vintage))  %>% 
#   summarise(Renewals=mean(Response), Count=sum(Count)) %>% 
#   as.data.frame()
# 
# rm(Vintages)
# data=data %>% select(-Vintage)

# cleaning the policy_sales_channel. 
# There is too many of these channels, they differ on renewal rate and volume
# we don't know anything specific about the sales channels which have low sales volume
Sales_channels=data %>% 
               # mutate(Response=as.integer(Response)-1) %>% 
               select(Policy_Sales_Channel,Response)  %>%  
               group_by(Policy_Sales_Channel) %>%
               mutate(Count=n_distinct(Policy_Sales_Channel))  %>% 
               summarise(Renewals=mean(Response), Count=sum(Count)) %>% 
               arrange(-Count,-Renewals) %>% 
               as.data.frame()

# Based on the density plot below and the above observations, I suggest propose 10 groups
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
# notice, that the renewal rate is 25% higher in case of high premium instances
k=52000
quantile(data$Annual_Premium ,probs = seq(0,1,by=0.05))

data %>% #mutate(Response=as.integer(Response)-1) %>% 
         select(Annual_Premium,Response) %>% 
         mutate(Indicator=(Annual_Premium>k)*1) %>% 
         group_by(Indicator) %>%
         summarise(mean=mean(Response))

# Create an additional column of values in excess of k 
# works more efficiently through use of indicator, than rowwise function
Premiums=data %>% 
  select(Annual_Premium) %>% 
  mutate(Indicator=(Annual_Premium>k)*1,
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
# Upsampling  ----
# ///////////////////////////////////////

# there is no point to do proper resampling because of the low number of 
# Response==1 cases and so, it might introduce too much noise 
train_data_resampled=rbind(train_data,
                           train_data %>% filter(Response==1))

# Random over sampling 
train_data_ROSE=ROSE(Response ~ ., data=train_data[1:100000,], seed=seed)$data

# syntchetic minority oversampling
# 
train_data_SMOTE=rbind(train_data,
                 SMOTE(form = Response ~ .,
                       data = train_data[1:100000,] %>% select(-id) %>% mutate(Response=factor(Response)),
                       perc.over = 100,
                       perc.under = 0,
                       k=5))


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

