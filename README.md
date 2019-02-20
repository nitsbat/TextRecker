# TextRecker
The project aims at building a machine learning model to filter mobile phone spam messages using Naive Bayes Classifier. The Dataset is in the repository titled as spam.csv. The model is built using R language.

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
 
 __All the same codes below are present in different R scripts. You can directly run the code from console by command :__
      >> Rscript main.R
      
## Introduction

The dataset i.e spam.csv on which we are performing the analysis contains only two fields which are type and text. The text field contains the SMS messages which are of string form i.e character while the type field is a factor containing two levels - **SPAM** and **HAM**.

## TEXT MINING
Text data mining involves combing through a text document or resource to get valuable structured information. This requires sophisticated analytical tools that process text in order to glean specific keywords or key data points from what are considered relatively raw or unstructured formats.The package 'tm' is available in R especially for text mining.

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



