rm(list = ls())
setwd("C:/Users/nits/Desktop/Analytics/R/TUTS/3 - Naive Bayes")
library(gmodels)
library(class)

train <- read.csv('spam.csv',stringsAsFactors = FALSE)
head(train,2)
str(train)
train$type <- factor(train$type)
table(train$type)
library(tm)

vignette("tm")

##Creating acorpus
sms_corp <- VCorpus(VectorSource(train$text))
print(sms_corp)
inspect(sms_corp[1:2])
as.character(sms_corp[[1]])
lapply(sms_corp[1:4],as.character)
##writeCorpus(sms_corp) - To save the corpus object on hard disk.

##USing the tm function for cleaning the data

sms_corp_clean <- tm_map(sms_corp,content_transformer(tolower)) ## to lower is a function
as.character(sms_corp[[1]])
as.character(sms_corp_clean[[1]]) ## checking the texts.

sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(removeNumbers))
as.character(sms_corp_clean[[4]])##checking the texts.


t = c('are','in')
sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(removeWords),t)
sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(removeWords),stopwords())


##removing the punctuatuion from the sentences
##It can be directly done by removePunctuation() but that will lead to unwanted consequences
##So making my own removePunctuatuion function in which the punctuation will be replaced by space.

removePunc <- function(x)
{
  gsub("[[:punct:]]+"," ",x) ##gsub actually substitutes a pattern from a sentence
}
removePunc("wew.?wqew:]sf.f")

sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(removePunc))
as.character(sms_corp_clean[[1]])
as.character(sms_corp_clean[[2]])

####### Stemming the words

library(SnowballC)
wordStem(c("learned","learning","learns"))

sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(stemDocument))
as.character(sms_corp_clean[[1]])

sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(stripWhitespace))

for (x in 1:10) {
  print(sms_corp_clean[[x]])
}

####### Tokenization
rm(sms_dtm)
sms_dtm <- DocumentTermMatrix(sms_corp_clean)
sms_dtm
#### to do the thing directly without preprocessing
sms_temp <- DocumentTermMatrix(sms_corp, control = list(tolower=TRUE,removeNumbers=TRUE,
                                                        stopwords = TRUE, stemming = TRUE,
                                                        removePunctuation = TRUE))

###### Data PReparation

sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]
sms_train_labels <- train[1:4169,]$type
sms_test_labels <- train[4170:5559,]$type
table(sms_test_labels)

###### Visulaisation

library(wordcloud)

wordcloud(sms_corp_clean , min.freq = 50,random.order = FALSE) # forms a wordcloud

spam <- subset(train, type=='spam') # forming subsets on the basis of type.
ham <- subset(train, type=='ham')

wordcloud(spam$text,max.words = 40,scale = c(5,0.2)) # max.words means the top 40 words
wordcloud(ham$text,max.words = 40,scale = c(3,0.5)) # scale gives the max and min size of font.

###### Data Preparation
#For applying the naive bayes in this text data we will have to prepare the data. 
#There are 6500 feature i.e words . It is unlikely that all are used for classification.
#To reduce the number of features, we will eliminate any word that appear in less than five SMS messages.

sms_freq <- findFreqTerms(sms_dtm_train,5) #this function will return a character vector in which there are words which are appearing atleast 5 times.
length(sms_freq)

#now we filter our DTM to contain only those words having most frequent words.
sms_dtm_freq_train <- sms_dtm_train[,sms_freq]
sms_dtm_freq_test <- sms_dtm_test[,sms_freq]

#The sparse matrix is in the form of 0 or 1. We have to change it to 'Yes' or 'No'
convert <- function(x)
{
  x <- ifelse(x>0,"Yes","No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,convert) # converting into Yes or No
sms_train[1:2,1:9] # MARGIN =1 specifies the rows while margin=2 specifies the columns
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,convert) # converting into Yes or No

#sms_temp <- apply(sms_dtm_freq_train, MARGIN = 1,convert)
#sms_temp[1:2,1:9]
library(e1071)
sms_classifier <- naiveBayes(sms_train,sms_train_labels) #the first parameter is train_data and second is the label
#and the third parameter is laplace which is discussed in naivebayes.txt
sms_pred <- predict(sms_classifier,sms_test) # the first parameter is the model classifier and second is the test data.
CrossTable(sms_pred,sms_test_labels,prop.chisq = FALSE) # The error is about 2.4% hence 97.6 % accuracy
#It means 30 out of 1390 are classified incorrectly.


sms_classifier2 <- naiveBayes(sms_train,sms_train_labels,laplace = 1)
sma_pred2 <- predict(sms_classifier2,sms_test)
CrossTable(sma_pred2,sms_test_labels,prop.chisq = FALSE)
# Here accuracy is increased as there are 27 out of 1390 are classified incorrectly.