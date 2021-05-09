# try SMOTE - “SMOTE: Synthetic Minority Over-sampling Technique“

# ///////////////////////////////////////
# Initialization ----
# ///////////////////////////////////////
# install_version("DMwR", "0.4.1") #Package ‘DMwR’ was removed from the CRAN repository.
# install_version("FCNN4R", "0.6.2") #Package ‘FCNN4R’ was removed from the CRAN repository.
library(tidyverse)
library(tidyquant)
library(caret)
library(pROC)
library(e1071)
library(glmnet)
library(remotes)
library(reshape2)
library(DALEX)
library(FCNN4R) # neural network with SGD and hyperparams
library(DMwR) # incl. SMOTE - Synthetic Minority Oversampling TEchnique - Nitesh Chawla, et al. 2002 
library(ROSE) # incl ROSE - Training and assessing classification rules with unbalanced data Menardi G.,Torelli N. 2013
library(doParallel)
library(data.table)
library(h2o) # despite h2o.init() output, h2o requires not the newest java (>16) but (8-15)

options(digits=2)
seed=100
set.seed(seed)

# Remember to run functions.R first

# Upload data
# Source: https://www.kaggle.com/arashnic/imbalanced-data-practice?select=aug_train.csv
# the testing dataset provided on the link mentioned above did not contain the actual values 
# "Response" is the target variable indicating whether a customer was successfully acquired
# "Policy_Sales_Channel" - insurance broker
data_initial=as.data.frame(read.csv("aug_train.csv")) %>% as_tibble() %>% arrange(id)
data=data_initial 

# there are no missing data
sum(is.na(data))==0
print(data)

# change data type to factors where necessary and apply binary encoding
data=data %>% mutate(Policy_Sales_Channel=as.factor(Policy_Sales_Channel),
                     Region_Code=as.factor(Region_Code),
                     # Driving_License=as.factor(Driving_License),
                     # Previously_Insured=as.factor(Previously_Insured),
                     Response=as.factor(Response), # it is more convenient to do that just before modeling
                     # Vehicle_Age=integer(Vehicle_Age),
                     Vehicle_Damage=(Vehicle_Damage=="Yes")*1,
                     Gender=as.factor(Gender)
                     ) %>% as.tibble()

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
# todos: correlations, visualization of car age/veh damage


# Distribution of the response variable by gender
data %>% ggplot(aes(x=Response))+
         geom_bar(aes(y = (..count..)/sum(..count..)))+
         scale_y_continuous(labels=scales::percent) +
         facet_grid(~Gender)+
         ylab("") +
         ggtitle("Distribution of the response variable",
                 subtitle = "It's an imbalanced dataset")

# Regions with the highest renewal rates by gender
data %>% 
  select(Region_Code,Response,Gender) %>% 
  mutate(Response=as.integer(as.character(Response))) %>% 
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
  mutate(High_Prem=(Annual_Premium>=55200)*1,
         Response=as.integer(as.character(Response))) %>% 
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
  mutate(Response=as.integer(as.character(Response))) %>% 
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
  mutate(Response=as.integer(as.character(Response))) %>% 
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
# there are 109 brokers (policy_sales_Channells) that have less than 
# K=150 customers
# their multitude is going to be handled later
table(data$Policy_Sales_Channel) %>% 
  as.data.frame() %>% 
  rename(Policy_Sales_Channel=1,Customers=2) %>% 
  arrange(-Customers)

# getting rid of brokers, which have less than K customers
K=150
data %>%
  mutate(Response=as.integer(as.character(Response))) %>% 
  group_by(Policy_Sales_Channel) %>% 
  summarise(Customers=n(),
            Renewals=mean(Response)) %>%  # arrange(-Customers) %>% print(n=100)
  filter(Customers>K) %>%
  ggplot(aes(x=reorder(Policy_Sales_Channel,Renewals),y=Renewals,fill=Customers))+
  geom_bar(stat="identity")+
  ylab("Renewal Rate") +
  xlab("Broker ID") +
  ggtitle("Brokers and renewal rates")

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

# Dependencies
# not suprisingly, the policy sales channels and regions are dependent
chisq.test(
  table(as.factor(data$Policy_Sales_Channel),
        as.factor(data$Region_Code)))

# Correlations
# correlations=cor(data %>% 
#                    mutate(Gender=(Gender=="Female")*1) %>% 
#                    select(Response,
#                           Gender,
#                           Driving_License,
#                           Previously_Insured,
#                           Vehicle_Damage,
#                           Annual_Premium,
#                           Vintage,
#                           Car_Age_New,
#                           Car_Age_Older,
#                           Car_Age_Oldest)) %>% 
#                   as.data.frame()

# clean the env
rm(Q,K)

# ///////////////////////////////////////
# data cleaning  ----
# ///////////////////////////////////////

# like mentioned above - lets separate out the tails of Vintage variable
data=data %>% 
      mutate(Vintage_Up=(Vintage>=285)*1,
                    Vintage_Down=(Vintage<=25)*1) %>% 
      select(-Vintage)

summary(data)

# cleaning the policy_sales_channel. 
# There is too many of these channels, they differ on renewal rate and volume
# we don't know anything specific about the sales channels which have low sales volume
Sales_channels=data %>% 
               mutate(Response=as.integer(as.character(Response))) %>% 
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
                 filter(Count>10) %>%
                 ggplot(aes(x=Renewals))+
                 geom_density()+
                 ggtitle("Density distribution of mean renewal rates",
                         subtitle = "Means calc. by sales channel")+
                 xlab("")+
                 ylab("")
  
# ///////////////////////////////////////
# feature engineering  ----
# ///////////////////////////////////////

# The classification is output as one-hot encoding 
data$Policy_Sales_Channel=data$Policy_Sales_Channel %>% as.matrix() %>% as.integer()
Sales_channels_one_hot=matrix(ncol=10,nrow=nrow(data))

# which brokers have volume of <= 75 clients (to put in the 1st group)
group_1=Sales_channels %>% 
        filter(Count<=75) %>% 
        select(Policy_Sales_Channel) %>% 
        as.matrix() %>% 
        as.integer()

# binary encoding of the first group
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

data %>% as.tibble()

Regions=predict(dummyVars("~Region_Code",data=data),
                newdata=data %>% as.data.frame) # check if data type (dbl) shouldn't be changed to int or fct

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

data %>% mutate(Response=as.integer(as.character(Response))) %>% 
         select(Annual_Premium,Response) %>% 
         mutate(Indicator=(Annual_Premium>k)*1) %>% 
         group_by(Indicator) %>%
         summarise(mean=mean(Response))

# Create an additional column of values in excess of k 
# works more efficiently through use of an indicator, than rowwise function
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
# train/test split and formatting----
# ///////////////////////////////////////
# data type handling
data=data %>% 
        select(-id) %>%
        mutate(Response=as.factor(Response),
               Gender=(Gender=="Female")*1) %>% 
        relocate(Response, .before=Gender) %>% 
        relocate(Annual_Premium,.before=Gender) %>% 
        relocate(Premium_Surplus,.before=Gender) %>% 
        relocate(Age,.before=Gender) %>% 
        mutate(across(c(5:76), as.integer)) %>% 
        as.tibble() 
        
# let's also sample the unchanged dataset for performance comparison
# onehotencoding and data type handling
data_initial=data_initial %>% 
             select(-id) %>%
             relocate(where(is.character)) %>% 
             relocate(Response, .before=Gender) %>% 
             relocate(Annual_Premium,.before=Gender) %>% 
             relocate(Vintage,.before=Gender) %>% 
             relocate(Age,.before=Gender) %>% 
             mutate(across(c(5:11),factor)) 

# one hot encoding 
data_initial = dummyVars("~ .", data=data_initial) %>% 
               predict(newdata = data_initial) %>% 
               as.tibble() %>% 
               mutate(across(c(5:224),as.integer)) %>% 
               mutate(Response=factor(Response))

# split
# note that for the initial data set we have to create a separate test set
# for the other cases (different resampling approaches) the other test set will be used
sample=c(sample_frac(as.data.frame(1:nrow(data)),size = 0.8))[[1]]
train_data=data[sample,]
test_data=data[-sample,]

train_data_initial=data_initial[sample,]
test_data_initial=data_initial[-sample,]

# ///////////////////////////////////////
# Upsampling  ----
# ///////////////////////////////////////
# There will be five datasets to test the different algorithms:
# 1. Initial data - the raw dataset, only incl. one-hot encoding of factor vars.
# 2. Base case - initial data after cleaning and feature engineering
# 3. Resampled - Base case including additional instances of the underpopulated observations 
# 4. ROSE - base case including additional instances generated through ROSE 
# 5. SMOTE - base case including additional instances generated through SMOTE 

# base case after cleaning and feature engineering
train_data %>% 
          group_by(Response) %>% 
          summarise(n=n()) %>%
          mutate(freq = n / sum(n))

# there is no point to do proper resampling because of the low number of 
# Response==1 cases and so, it might introduce too much noise 
train_data_resampled=rbind(train_data,
                           train_data %>% 
                             filter(Response==1))

# 71%-29%
train_data_resampled %>% 
          group_by(Response) %>% 
          summarise(n=n()) %>%
          mutate(freq = n / sum(n))

# Random over sampling 
# we have to correct for non-realistic outputs (negative premiums, or premiums lower than 2630) and 
# outputs that break the previously outlined "rules" (eg. annual premium can't exceed 52000)
# note - ROSE generates an entirely synthetic dataset
# note - compared to the 'resampled' case above, this dataset is almost twice as large
# note - the binary variables have to be converted to factors, because otherwise the ROSE function, 
# returns doubles and then we reconvert them to int
train_data_ROSE=rbind(train_data,
                      ROSE(Response ~ ., 
                           data=train_data %>%   
                                mutate(across(c(1,5:76), factor)),
                           seed=seed)$data %>% 
                      mutate(Annual_Premium=Normalize(Annual_Premium),
                             Premium_Surplus=Normalize(Premium_Surplus),
                             Age=Normalize(Age),
                             across(5:76,FactorToInteger)))

# Todo: SMOTE
# the SMOTE function takes forever to execute. Since the package is officially archived, 
# I might redo this algorithm 
# train_data_SMOTE=rbind(train_data,
#                  SMOTE(form = Response ~ .,
#                        data = train_data[1:1000,],
#                        perc.over = 100,
#                        perc.under = 0,
#                        k=5))

# ///////////////////////////////////////
# Dimension reduction  ---- 
# ///////////////////////////////////////
# rotation matrices from PCA
PCA_Rot_train_data_initial = prcomp(train_data_initial[,-1])
PCA_Rot_train_data = prcomp(train_data[,-1])
PCA_Rot_train_data_resampled = prcomp(train_data_resampled[,-1])
PCA_Rot_train_data_ROSE = prcomp(train_data_ROSE[,-1])


# Training sets
# k - how many principal components to keep
k=50

PCA_train_data_initial = train_data_initial[,-1] %>%
                         data.matrix() %*%
                         PCA_Rot_train_data_initial$rotation %>%
                         as_tibble() %>%
                         select(1:k) %>%
                         add_column(train_data_initial[,1],.before="PC1")

PCA_train_data = train_data[,-1] %>% 
                 data.matrix() %*% 
                 PCA_Rot_train_data$rotation %>% 
                 as_tibble() %>% 
                 select(1:k) %>% 
                 add_column(train_data[,1],.before="PC1")
  
PCA_train_data_resampled = train_data_resampled[,-1] %>% 
                           data.matrix() %*% 
                           PCA_Rot_train_data_resampled$rotation %>% 
                           as_tibble() %>% 
                           select(1:k) %>% 
                           add_column(train_data_resampled[,1],.before="PC1")

PCA_train_data_ROSE = train_data_ROSE[,-1] %>% 
                      data.matrix() %*% 
                      PCA_Rot_train_data_ROSE$rotation %>% 
                      as_tibble() %>% 
                      select(1:k) %>% 
                      add_column(train_data_ROSE[,1],.before="PC1")

# The first two principal components seem to have 'found' the way
# to characterize the Response.
# A similar profile is seen in each dataset case, except for the initial data.
# It seems that the feature engineering indeed helped to make differences
# between the two groups more distinguishable.
# Comparing the ROSE and resampled data that, ROSE does not introduce noise. 
PCA_train_data_initial %>%
  ggplot(aes(x=PC1,y=PC2,color=Response))+
  geom_point(alpha=0.5)+
  ggtitle("Principal Components by Response",
          subtitle="Initial Train Data")

PCA_train_data %>% 
  ggplot(aes(x=PC1,y=PC2,color=Response))+
  geom_point(alpha=0.5)+
  ggtitle("Principal Components by Response",
          subtitle="Train Data")

PCA_train_data_resampled %>% 
  ggplot(aes(x=PC1,y=PC2,color=Response))+
  geom_point(alpha=0.5)+
  ggtitle("Principal Components by Response",
          subtitle="Resampled Train Data")

PCA_train_data_ROSE %>% 
  ggplot(aes(x=PC1,y=PC2,color=Response))+
  geom_point(alpha=0.5)+
  ggtitle("Principal Components by Response",
          subtitle="ROSE Train Data")

# Test sets 
# I apply the train set rotation matrices to create new train sets.
# The basic assumption of any train-test setting, is that 
# the hold-out dataset has the same distribution as the train set. 
# However, it should be checked anyway if the formulation of principal components
# of the test set does not differ significantly. 
# Similarly, these distributions should not change upon resampling/ROSE but
# to rule out the chance for errors due to that, I create separate test sets
PCA_test_data_initial = test_data_initial[,-1] %>%
                        data.matrix() %*%
                        PCA_Rot_train_data_initial$rotation %>%
                        as_tibble() %>%
                        select(1:k) %>%
                        add_column(test_data_initial[,1],.before="PC1")

PCA_test_data = test_data[,-1] %>% 
                data.matrix() %*% 
                PCA_Rot_train_data$rotation %>% 
                as_tibble() %>% 
                select(1:k) %>% 
                add_column(test_data[,1],.before="PC1")

PCA_test_data_resampled = test_data[,-1] %>% 
                          data.matrix() %*% 
                          PCA_Rot_train_data_resampled$rotation %>% 
                          as_tibble() %>% 
                          select(1:k) %>% 
                          add_column(test_data[,1],.before="PC1")

PCA_test_data_ROSE = test_data[,-1] %>% 
                     data.matrix() %*% 
                     PCA_Rot_train_data_ROSE$rotation %>% 
                     as_tibble() %>% 
                     select(1:k) %>% 
                     add_column(test_data[,1],.before="PC1")

# clean env - store train and test objects in lists and delete global variables
TRAIN=list(data_initial=train_data_initial,
           data=train_data,
           data_resampled=train_data_resampled,
           data_ROSE=train_data_ROSE,
           
           PCA_data_initial=PCA_train_data_initial,
           PCA_data=PCA_train_data,
           PCA_data_resampled=PCA_train_data_resampled,
           PCA_data_ROSE=PCA_train_data_ROSE)

TEST=list(data_initial=test_data_initial,
          data=test_data,
          
          PCA_data_initial=PCA_test_data_initial,
          PCA_data=PCA_test_data,
          PCA_data_resampled=PCA_test_data_resampled,
          PCA_data_ROSE=PCA_test_data_ROSE)

rm(k,
   
   train_data_initial,
   train_data,
   train_data_resampled,
   train_data_ROSE,
   
   test_data_initial,
   test_data,

   PCA_Rot_train_data_initial,
   PCA_Rot_train_data,
   PCA_Rot_train_data_resampled,
   PCA_Rot_train_data_ROSE,
   
   PCA_train_data_initial,
   PCA_train_data,
   PCA_train_data_resampled,
   PCA_train_data_ROSE,   
   
   PCA_test_data_initial,
   PCA_test_data,
   PCA_test_data_resampled,
   PCA_test_data_ROSE)

# ///////////////////////////////////////
# Modeling - setting up ---- 
# ///////////////////////////////////////
# parallel processing
registerDoParallel(makePSOCKcluster(4))

trControl = trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 2)

# prep objects to store results
Accuracy=matrix(nrow=5,ncol=5,NA) %>% as.data.frame()
colnames(Accuracy)=c("Initial",
                     "Basic",
                     "Resampled",
                     "ROSE",
                     "SMOTE")  

rownames(Accuracy)=c("LogisticRegression",
                     "NeuralNetwork",
                     "DecisionTree",
                     "RandomForest",
                     "GBM")

loopnames=c("PCA_data_initial",
            "PCA_data",
            "PCA_data_resampled",
            "PCA_data_ROSE")

Specificity=Accuracy

ConfMat=list()
Models=list()
Predictions=list()

# ///////////////////////////////////////
# Modeling ------
# ///////////////////////////////////////
# indicate how many observations to take from the training set
# to initally evaluate the models. Can be set to nrow(data) but
# then the training will take much longer.
# for now it takes the first n=SubSet observations, later do: random sampling
SubSet=10000
  
  rm(Predict_Logistic_reg_fact,
     Predict_Logistic_reg_Initial_fact,
     Predict_Logistic_reg_resampled_fact,
     Predict_Logistic_reg_ROSE_fact)

# First I fit a simple model on the available data sets 
# (Disregard the warnings relating to row names in tibbles)
Best_LR=list()
treshold=0.5

for (i in 1:4){
  print(paste("Training model: Logistic Regression, data: ", loopnames[i], sep=""))
  
  training_frame = TRAIN[[loopnames[i]]] %>% slice(1:10000)
  validation_frame=TEST[[loopnames[i]]] %>% slice(1:10000)
  
  # fit logistic regression
  temp=train(x=training_frame %>% select(-1) ,
                              y=training_frame$Response,
                              method = 'glmnet',
                              trControl = trControl,
                              family = 'binomial' )
  
  Best_LR[loopnames[i]]=list(temp)
  # confusion matrix for test data
  Prediction=as.factor((predict(temp,
                                validation_frame %>% select(-1),
                                type = "prob")[,2] > treshold)*1)
  
  Reference=validation_frame$Response %>% factor()
  
  CM=confusionMatrix(data=Prediction,reference=Reference)
  ConfMat[[paste("LR_Test",loopnames[i],sep="_")]]=CM

  # confusion matrix for train data
  Prediction=as.factor((predict(temp,
                                training_frame %>% select(-1),
                                type = "prob")[,2] > treshold)*1)
  
  Reference=training_frame$Response %>% factor()
  
  CM=confusionMatrix(data=Prediction,reference=Reference)
  ConfMat[[paste("LR_Train",loopnames[i],sep="_")]]=CM  
  
  # delete all not needed models and clear temporary grid
  rm(Prediction,
     Reference,
     CM,
     training_frame,
     validation_frame,
     temp)
  
}

# h2o models ----
h2o.init()
h2o.init(nthreads=3, max_mem_size="2G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

loopnames=c("PCA_data_initial",
            "PCA_data",
            "PCA_data_resampled",
            "PCA_data_ROSE")

# Neural Networks ----
# these models are trained in a loop with hyperparameters found through
# grid search, which is performed on a fraction of the dataset.
# otherwise, one iteration took over an hour.
# After finding the optimal hyperparameters on a subsample of the training set,
# I fit the network on the full training data (in each data case)

# to store parameters of the best models
Best_NN=list()

param_tune_NN= list(
  # hidden=list(c(50,50,10),c(40,30,10),c(40,30,5)),
  rate=c(0.09,0.08,0.07,0.06),
  momentum_stable=c(0.9,0.8,0.7),
  momentum_start=c(0.9,0.8,0.7)
)

for (i in 1:4){
  print(paste("Grid Search of params: NN, data: ", loopnames[i], sep=""))
  
  training_frame = as.h2o(TRAIN[loopnames[i]] %>%
                            as.data.frame() %>%
                            slice(1:100000),
                          use_datatable = FALSE)
  
  validation_frame=as.h2o(TEST[loopnames[i]] %>%
                            as.data.frame() %>%
                            slice(1:10000),
                          use_datatable = FALSE)
  
  # variable referencing in the h2o.grid object
  ColNames=colnames(training_frame)
  
  Best_NN[loopnames[i]]=h2o.grid(
                    algorithm="deeplearning",
                    hyper_params=param_tune_NN,
                    grid_id=paste(loopnames[i],"NN",sep="_"), 
  
                    # Data inputs
                    training_frame = training_frame,
                    validation_frame=validation_frame,
                    x=setdiff(ColNames, ColNames[1]),
                    y=ColNames[1],
                    
                    # Basic hyperparams.
                    hidden=c(40,30,10),
                    activation="Tanh",   ## for ReLU I encountered the exploding gradient problem 
                    epochs=50,
                    # mini_batch_size = 10000,
                    
                    adaptive_rate=F,
                    rate_annealing=0.1^(7),  # picked after lots of trials 
                    
                    # Manual learning parameters 
                    # score_validation_samples=1000,
                    # rate=0.05,
                    # rate_annealing=2e-6,            
                    # momentum_start=0.5,
                    # momentum_stable=0.9,          
                    
                    
                    # Output
                    variable_importances=T,    ## not enabled by default
                    initial_weight_distribution = "Normal",
                    seed = seed
                    # verbose = TRUE,
                    # reproducible=TRUE,
  
                    # Early Stopping
                    # stopping_metric="MSE", ## could be "MSE","logloss","r2"
                    # stopping_tolerance=0.001,
  )
  
  # to keep results from all models comparable, I translate the h2o object
  # back to a regular vector but even though as.data.frame() conversion
  # outputs a factor, I needed to do some equilibristics to get confusionMatrix
  # function to accept the inputs 
  
  # first I retrieve the list of models resulting form grid search
  grid=h2o.getGrid(grid_id = paste(loopnames[i],"NN",sep="_"),
                   sort_by="roc",decreasing=FALSE)
  
  # print(grid)
  
  BestModel=h2o.getModel(grid@model_ids[[1]])
  
  # overwrite the grid object with the best models' parameters
  Best_NN[loopnames[i]]=list(BestModel@model[["model_summary"]])
  
  # make the prediction on the best model (Testing set)
  Prediction=h2o.predict(BestModel,
                         newdata=validation_frame)[,1]%>%
                                 as.data.frame()  %>%
                                 as.matrix() %>%
                                 factor()
  
  Reference=validation_frame[,1] %>% 
            as.data.frame() %>% 
            as.matrix() %>% 
            factor()
  
  # feed the Prediction and true values to the caret confusionMatrix and save
  CM=confusionMatrix(data = Prediction,reference = Reference)
  ConfMat[[paste("NN_Test",loopnames[i],sep="_")]]=CM
  
  # make the prediction on the best model (Training set)
  Prediction=h2o.predict(BestModel,
                         newdata=training_frame)[,1]%>%
                         as.data.frame()  %>%
                         as.matrix() %>%
                         factor()
  
  Reference=training_frame[,1] %>% 
            as.data.frame() %>% 
            as.matrix() %>% 
            factor()
  
  # feed the Prediction and true values to the caret confusionMatrix and save
  CM=confusionMatrix(data = Prediction,reference = Reference)
  ConfMat[[paste("NN_Training",loopnames[i],sep="_")]]=CM
  
  # delete all not needed models and clear temporary grid
  h2o.removeAll()
  rm(grid,
     Prediction,
     Reference,
     training_frame,
     validati_frame,
     BestModel)
}



# Random Forests ----
# Same approach as for deep learning
Best_RF=list()

treshold=0.6

# variable referencing in the h2o.grid object
param_tune_RF= list(max_depth = c(20,25,30),
                    min_rows = c(5,7,10),
                    ntrees = c(40,50))

for (i in 1:4){
  print(paste("Grid Search of params: RF, data: ", loopnames[i], sep=""))
    
  # subset of training frame in h2o format
  training_frame = as.h2o(TRAIN[loopnames[i]] %>%
                            as.data.frame() %>%
                            slice(1:100000),
                          use_datatable = FALSE)
  
  # subset of testing frame in h2o format
  validation_frame=as.h2o(TEST[loopnames[i]] %>%
                       as.data.frame() %>%
                       slice(1:10000),
                     use_datatable = FALSE)
  
  ColNames=colnames(validation_frame) 
  
  # grid search for best model
  Best_RF[loopnames[i]]=h2o.grid(
                        algorithm="randomForest",
                        hyper_params=param_tune_RF,
                        grid_id=paste(loopnames[i],"RF",sep="_"),
  
                        training_frame = training_frame,
                        validation_frame= validation_frame,
                        
                        x=setdiff(ColNames, ColNames[1]),
                        y=ColNames[1],
                        seed = seed)

  # to keep results from all models comparable, I translate the h2o object
  # back to a regular vector but even though as.data.frame() conversion
  # outputs a factor, I needed to do some equilibristics to get confusionMatrix
  # function to accept the inputs 
  
  # first I retrieve the list of models resulting form grid search
  grid=h2o.getGrid(grid_id = paste(loopnames[i],"RF",sep="_"),
                   sort_by="auc",decreasing=FALSE)
  
  # print(grid)
  
  BestModel=h2o.getModel(grid@model_ids[[1]])
  
  # overwrite the grid object with the best models' parameters
  Best_RF[loopnames[i]]=list(BestModel@model[["model_summary"]])

  # Confusion matrix for the test set
  Prediction=h2o.predict(BestModel,
                         newdata=validation_frame)[,1] %>%
                         as.data.frame()  %>%
                         as.matrix() %>%
                         factor()

  Reference=validation_frame[,1] %>% 
            as.data.frame() %>% 
            as.matrix() %>% 
            factor()
  
  # feed the Prediction and true values to the caret confusionMatrix and save
  CM=confusionMatrix(data = Prediction,reference = Reference)
  ConfMat[[paste("RF_Test",loopnames[i],sep="_")]]=CM
  
  # Confusion matrix for the training set
  Prediction=h2o.predict(BestModel,
                         newdata=training_frame)[,1] %>%
                         as.data.frame()  %>%
                         as.matrix() %>%
                         factor()
  
  Reference=training_frame[,1] %>% 
            as.data.frame() %>% 
            as.matrix() %>% 
            factor()
  
  # feed the Prediction and true values to the caret confusionMatrix and save
  CM=confusionMatrix(data = Prediction,reference = Reference)
  ConfMat[[paste("RF_Training",loopnames[i],sep="_")]]=CM
  
  # delete all not needed models and clear temporary grid
  h2o.removeAll()
  rm(grid,
     Prediction,
     Reference,
     training_frame,
     validati_frame,
     BestModel)
  
}





# turn of parallel computing
stopCluster(makePSOCKcluster(4))

# ///////////////////////////////////////
# XAI ---- 
# ///////////////////////////////////////

# Prepare model explainer
explainer = explain(Logistic_reg,
                     data  = train_data,
                     y     = as.integer(as.character(train_data$Response)),
                     label = "Logistic Regression on basic dataset"
)

# variable importance

explainer %>% 
  model_parts(loss_function = loss_default(explainer$model_info$type),
              type = "variable_importance",
              N = n_sample,
              n_sample = 1000
              ) %>% 
  plot()  

# shapley values
explainer %>% 
  predict_parts(new_observation = test_data %>% 
                                  filter(Response==1) %>% 
                                  sample_n(4),
                type = "shap") %>% 
  plot()





explainer %>% 
  model_performance() %>% 
  plot(geom="roc")

# ///////////////////////////////////////
# notes ----
# ///////////////////////////////////////


# check if the dependency eg on age is linear and whether there is a significant difference among two groups
# if yes - then add +/-1 to the age (thus create two extra data instances that bought the policy)

# Pre-processing - PCA, LDA/QDA + Logistic Regression, NN, RF

# Neural network -> logistic regression (like suggested by Mario Wutrich)

# Explainability

