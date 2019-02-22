# TextRecker
The project aims at building a machine learning model to filter mobile phone spam messages using Naive Bayes Classifier. The Dataset is in the repository titled as [spam.csv](spam.csv). The model is built using R language.

## Software Requirements
 #### R-Studio - https://www.rstudio.com/products/rstudio/download/
 #### The following packages need to be installed in order to perform text mining and Naive Bayes.
 * _gmodels_ - for table() and Crosstable() functions.
 * _tm_ - text mining package.
 * _SnowballC_ - for word stemming
 * _wordcloud_ - for visualisation of texts.
 * _e1071_ - It contains the Naive bayes classifier.
 
 **The basic syntax to download and then load the packages is -** 

      > install.packages(gmodels)
      > library(gmodels) #this command is used to load the package if already installed   
 
 ***************************************************************************************************************
 
 __All the same codes below are present in [main.R](main.R). You can directly run the code from console by command :__
      >> Rscript main.R


 ***************************************************************************************************************
## Introduction

The dataset i.e spam.csv on which we are performing the analysis contains only two fields which are type and text. The text field contains the SMS messages which are of string form i.e character while the type field is a factor containing two levels - **SPAM** and **HAM**.

## TEXT MINING
Text data mining involves combining through a text document or resource to get valuable structured information. This requires sophisticated analytical tools that process text in order to glean specific keywords or key data points from what are considered relatively raw or unstructured formats.The package 'tm' is available in R especially for text mining.

1. The first step in text processing is to create a corpus which means a collection of text documents. In our model it will be a collection of SMS messages.
To see the detail use of "tm" package type ` vignette("tm").` It will open a help guide pdf.
We will be creating a VCorpus i.e Volatile Corpus in which the object is stored in memory not on the disk. Once released the whole object gets destroyed.
```
      sms_corp <- VCorpus(VectorSource(train$text))    
      print(sms_corp)
```     
VectorSource is a source object made from predefined sources like DirSource, VectorSource, or DataFrameSource.It only accepts the vector i.e character vector. The print statement will show the following output. It shows that the total number of documents made are 5559.
```
<<VCorpus>>
Metadata:  corpus specific: 0, document level (indexed): 0
Content:  documents: 5559
```

To inspect a single document one can use `inspect(sms_corp[1:2])`. It shows the following output which you can easily interpret.
```
<<VCorpus>>
Metadata:  corpus specific: 0, document level (indexed): 0
Content:  documents: 2

[[1]]
<<PlainTextDocument>>
Metadata:  7
Content:  chars: 49

[[2]]
<<PlainTextDocument>>
Metadata:  7
Content:  chars: 23
```

To see the text of particular document use `as.character(sms_corp[[1]])`. Also we can save the Corpus permanently in our hard disk by using command `WriteCorpus(Corpusname)`

2. The second step in text mining includes cleaning of the raw text which means to remove the punctuations and unwanted characters.The function use to apply this transformation is tm_map(). We will use this function to clean up our corpses and save the new transformation in new corpus. tm_map is genearally used as the lapply function but the difference is it takes the the corpus while lapply function takes the list as a parameter. It is a very important function with syntax as -

      `tm_map(corpusname,function_name).`

The function_name i.e second parameter can be the functions which are inbuilt in "tm" package only. If need to use the function other than functions of "tm" package, use the content_transformer function with the inbuilt function inside it. Here content_transformer acts as to transform overall corpus and cleanup process such as grep pattern matching.So initially our first task will be to convert the whole text to lowercase.

- For making the text to get converted into lowercase. We will use tolower() function . 
```
      sms_corp_clean <- tm_map(sms_corp,content_transformer(tolower))
      as.character(sms_corp[[1]])
```
Next we will remove the stopwords() . Stop words are basically the words like to, and, but, for, each , at, etc which are generally used in text but are not useful in machine learning. To remove the stopwords use the stopwords() in the tm package. We can make our own list of stopwords() and then remove it. Though stopwords() is the function in tm package only but it actually returns the vector containing the stopwords. Hence it should be used with the removeWords() function and in second parameter the words we need to remove. Hence we will use stopwords with removeWords.
```
t = c('are','in') ## by this we can remove the word 'are' and 'in'.
sms_corp_clean <- tm_map(sms_corp_clean,removeWords,t)
sms_corp_clean <- tm_map(sms_corp_clean,removeWords,stopwords()) ## by this we can remove all the stopwords().
```

3. Next step is to remove the punctuation marks but remove punctuation marks directly leads to unwanted consequences. e.g
 `removePunctuation("we..are?:hackers")` results in "wearehackers". We can see it just strips the punctuations and concat the rest of it.

Better to make our own removePunctuation() function. i.e
```
removePunc <- function(x)
{
  gsub("[[:punct:]]+"," ",x) ##gsub actually replaces a pattern from a sentence by whitespace.
}
removePunc("wew.?wqew:]sf.f") ## It will give the result as "wew wqew sf f"
sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(removePunc)) #Applying the function to the corpus 
```

Here the gsub function actually replaces the given string or pattern with a user string from a sentence or text. The first parameter of the function is the pattern i.e regular expression and the second parameter is what to replace i.e desired text and the third one is from which sentence.

4. Next step is to reduce the given words to its root. e.g - Words like learning, learns, learned can all be reduced to its root word learn. This process is called stemming. The above words are stripped such that the suffix are stripped to transform them into base form. 
Stemming is down by the package `SnowballC`, it can be easily integrated with tm package.
**wordStem()** function reduces the words to its base form and return a character vector. To apply this wordstem to full Corpus we will use `stemDocument()` function.

`wordStem(c("learned","learning","learns"))`
`sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(stemDocument))`
`as.character(sms_corp_clean[[2]])`

The following code will give an output like - `"k give back thank"` . Here thanks is reduced to thank.

5. Last step in data cleaning is to remove the extra whitespaces using stripwhitespaces() function.

`sms_corp_clean <- tm_map(sms_corp_clean,content_transformer(stripWhitespace))`

***********************************************************************************************************************************

- The final step in text mining is the most important i.e -  to split the messages into individual units called tokens and the process is called tokenization. A token is the single element of a text string , in this case they are words.
The tm package uses **DTM()** function to tokenize the string. *DocumentTermMAtrix()* function will take the corpus and create a data structure called DTM in which rows indicates the message(represented by message id) while the columns indicate the terms or words.
Similarly it also has the TDM i.e TermDocumentMatrix which is a transpose of DTM. It is favourly used when number of documents is small and word length is large.

 `sms_dtm <- DocumentTermMatrix(sms_corp_clean)`


This will result in creation of a sparse matrix in which the cell stores a number indicating a count of the times the word represented by the column appears in the document represented by the row. The vast majority of cells in the matrix is filled with zeros. 
If the above command is giving error try to execute all the above commands with content_transformer(). On the other hand if you hadn't done any preprocessing we can do it by providing a list of parameters.
```
 sms_temp <- DocumentTermMatrix(sms_corp, control = list(tolower=TRUE,removeNumbers=TRUE,
                                                        stopwords = TRUE, stemming = TRUE,
                                                        removePunctuation = TRUE))
```

The result of this is little bit different due to the stopwords() function. Some words split differently when they are cleaned before tokenization. After the tokenization we can analyse and make a model using this sms_dtm. One can apply a regular modulation in sms_dtm.

***********************************************************************************************************************************

# Data Visualisation
We will use Wordcloud for this. Wordcloud is a way to depict the frequency at which words appear in text data. the cloud is composed of scatter words in which the most appearing words are shown in bold and large font while the one less appearing is shown at small font size. we will install wordcloud package.

`wordcloud(sms_corp_clean , min.freq = 50,random.order = FALSE)`

![rplot](https://user-images.githubusercontent.com/22686274/53234512-afec7300-36b5-11e9-9771-2fb474dbe981.png)
```
spam <- subset(train, type=='spam') # forming subsets on the basis of type.
ham <- subset(train, type=='ham')
wordcloud(spam$text,max.words = 40,scale = c(5,0.2)) # max.words means the top 40 words
wordcloud(ham$text,max.words = 40,scale = c(3,0.5)) # scale gives the max and min size of font.
```
Output for Spam Subset will be shown like this - 
![rplot01](https://user-images.githubusercontent.com/22686274/53234658-0b1e6580-36b6-11e9-8e98-ca83dcefc1cd.png)

**************************************************************************************************************************************

# Naive Bayes 
Naive Bayes is based on bayes probabiltiy in which the likelihood of an event is based on the evidence and across multiple trials.
To really know how the algorithm works [check](https://www.geeksforgeeks.org/naive-bayes-classifiers/). and to understand with example [check](https://drive.google.com/open?id=1FB4_J78cMhxp8BAMTVtDc3NoY9fWbXlc) 

To apply the Naive Bayes we will have to first prepare the data for it. There are 6500 feature i.e words . It is unlikely that all are used for classification. To reduce the number of features, we will eliminate any word that appear in less than five SMS messages. We filter our DTM to contain only those words having most frequent words.
```
sms_dtm_freq_train <- sms_dtm_train[,sms_freq]
sms_dtm_freq_test <- sms_dtm_test[,sms_freq]

#The sparse matrix is in the form of 0 or 1. We have to change it to 'Yes' or 'No' coz Naive Bayes is performed on categorical data.
convert <- function(x)
{
  x <- ifelse(x>0,"Yes","No")
}
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,convert) # converting into Yes or No , Margin=2 specifies the columns
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,convert) # converting test data into Yes or No
```
## Applying Naive Bayes

```

library(e1071)
sms_classifier <- naiveBayes(sms_train,sms_train_labels) #the first parameter is train_data and second is the label
sms_pred <- predict(sms_classifier,sms_test) # the first parameter is the model classifier and second is the test data.
CrossTable(sms_pred,sms_test_labels,prop.chisq = FALSE)
```

The error is about 2.4% hence **97.6 %** accuracy
It means 30 out of 1390 are classified incorrectly.

:metal: *Code for Naive Bayes is present in [main.r](main.R)*

---------
#                THANK YOU :v: :raising_hand:

