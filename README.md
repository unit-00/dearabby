# Dear Abby
NLP Analysis of Dear Abby column

## Table of Contents
- [Background](#background)
- [Motivation](#motivation)
- [Data](#data)
    - [Understanding](#understanding)
    - [Preparation](#preparation)
- [Unsupervised Modelling](#unsupervised-modelling)
    - [KMeans](#kmeans)
    - [NMF](#nmf)
    - [LDA](#lda)
- [Application](#application)
- [Conclusion](#conclusion)
- [Follow Up](#follow-up)
- [Credit](#credit)
- [Reproduce Model](#reproduce-model)

## Background
[Dear Abby](https://en.wikipedia.org/wiki/Dear_Abby) is an advice column founded in 1956 by Pauline Phillips under the pen name "Abigal Van Buren" where now her daughter carry her mantle. 

People write in to hear advice from Abby regarding issues they're having in life.

In terms of the directory structure of the project
- data 
    - script to collect the data and the data itself
- image
    - visualization for the readme
- model 
    - script to reproduce the result 
    - notebooks for model visualization
    - model used for this project

## Motivation
People are asking for advice on a range of troubles, I wanted to see what are the topics are people asking for advice on.

## Data
I have collected daily articles from [Dear Abby](https://www.uexpress.com/dearabby) column, dating from May 1991 to May 2020.

From initial collection, the source files were then thrown into a MongoDB warehouse for further processing.

Please take a look at `collect_data.py` to see how the data was collected. The script was built with `requests`, `BeautifulSoup4`, `pandas`, and `MongoDB`.

### Understanding
Each day consists of text in a question and answer format, having 1 to 3 questions a day. Resulting to about ~21000 questions. Each piece of text written in English without emojis or internet slang.

In addition starting from 2012, multiple categories were assigned to each question, and that was collected as well. However due to a lack of data prior to 2012, this resulted in about 68% of the feature missing.

I made a decision to use the 32% of the categorized data to model, to verify if the topics generated seemed to cluster well. 

During exploration, I have found that majority of the text have a distribution of about normal centered around 1000 words, with a bit of a right tail. Shown in the graph below.

![mean char length](/image/word_length.jpg)

Furthermore I have found that the mean character length of each question and answer has distribution of about normal centered around 4.5 characters, with a long left tail. Shown in the graph below.

![mean char length](/image/char_length.jpg)

### Preparation
To prepare the data for modelling,

The preprocessing steps
- DEAR [ABBY] or DEAR [WRITER] were removed
- Text Tokenized
- Encodings were removed
- Text transformed into lowercase
- Stopwords were removed
- Text stemming
- Remove words less than 3 characters
- Made into unigram, bigram, and trigrams

Each text then had features extracted via both document frequency and term frequency - inverse document frequency (tfidf) methods capping at 2000 features.

Because of the high dimension of this feature matrix, I then had do dimension reduction with `PCA` before to mitigate curse of dimensionality.

## Unsupervised Modelling
I used three algorithms to do topic modelling
- KMeans
- Non-matrix Factorization (NMF)
- Latent Dirichlet Allocation (LDA)

### KMeans
KMeans is an algorithm that clusters by trying to find the minimum distance of each points to a center. This was done on the reduced dimension version of the tfidf matrix.

In order to decide on the number of clusters, I used the silhouette method.  

![silhouette tfidf](/image/silhouette_tfidf.jpg)

Since it peaked at 3 clusters, the topic modelling was done with kmeans of n = 3.

![kmeans tfidf](/image/kmeans_tfidf.png)

With the nice slope of the three clusters in the silhouette plot, we see this reflecting in the clustered data where the clusters are nicely divded.

For the full result, please check out the [eda](/model/eda.ipynb) notebook.

Taking a closer look at the words being used for each clusters. 


We can see that this cluster might be more about gifts, gratuity, and working out some sort of issue.

```
Cluster 0:
friend gift would peopl like think invit person know feel want ask make wed famili tell someon thank work take
```

As for this cluster we can see that this is more about dating, relationship, marriage.

```
Cluster 1:
friend want feel year love know relationship husband like would tell help think date talk make life work marri wife
```

For the third cluster, interestingly, even though they share some words, the context is shown to be different. With this cluster, it's more about family and marriage. 

```
Cluster 2:
husband daughter mother famili children want parent year would father sister live wife feel child marri know love help think
```

Now comparing it to the top 3 categories of each cluster. They seem to be around the same topics.

```
Topic 0:
Etiquette & Ethics     1090
Family & Parenting      769
Friends & Neighbors     512

Topic 1:
Love & Dating         906
Marriage & Divorce    708
Family & Parenting    487

Topic 2:
Family & Parenting    1474
Marriage & Divorce     544
Money                  261
```

### NMF
Non-Negative Matrix Factorization (NMF) can find topics via latent features. It is done on the tfidf matrix to find the groupings. In order to find the optimal number of latent features, I used the elbow method on the marginal difference.

Shown in the graph below. 

We can see that at 4 components, we see a sharp drop.

![recon err nmf](/image/recon_err_nmf.jpg)

These were the words associated with each component.  

We can see this topic is associated with family.
```
Topic #0: 
daughter mother parent children famili father child sister live babi brother kid want home help girl would know care visit
```

This second one seems to be more about gifts and celebration.
```
Topic #1: 
gift wed invit parti birthday card would guest thank send attend friend famili dinner receiv celebr give shower note christma
```

As for the third one, it seems to have to do with relationships and dating.
```
Topic #2: 
friend feel like peopl know want date relationship person think would tell make someon talk work boyfriend help love thing
```

The last one being more about marriage.
```
Topic #3: 
husband wife marriag marri year love divorc want togeth feel famili work say counsel would never affair know help think
```
We can see that NMF is able to develop distinct topics with each latent features. Comparing it with the aggregated categories based on this clusters, similarly to KMeans, it seems the keywords are on point.

```
Topic 0:
Family & Parenting    1266
Health & Safety        223
Money                  187

Topic 1:
Etiquette & Ethics         410
Holidays & Celebrations    389
Family & Parenting         304

Topic 2:
Love & Dating         968
Family & Parenting    679
Etiquette & Ethics    670

Topic 3:
Marriage & Divorce    929
Family & Parenting    481
Money                 165
```

### LDA
Last algorithm I used to model topics is the Latent Dirichlet Allocation (LDA). LDA is a model that generate topics based on the probability of the topic being in a document and the probability of the word being in a topic. 

Unlike KMeans and NMF, it doesn't have a rigorous method to narrow down how many topics there could be. Thus using results I've found earlier as a starting point, trial and errors were done.

The following is the keywords of each topic through LDA.

This one seems to be about relationship and parents.

```
Topic #0: help know feel want parent like need talk work tell would friend make school think year peopl life relationship find
```

As for this topic, it might be some along the lines of etiquette and maybe relationship as well.
```
Topic #1: would like think peopl ask gift friend husband person want make tell call know feel money thank someth someon look
```

Lastly, this seems more centered around family and marriage.
```
Topic #2: husband want famili year would feel friend live marri wife know love mother like daughter think home togeth make relationship
```

Interestingly even though the topics share the same keywords, the clustering methods are able to pick apart the context. 

Comparing it with the top 3 categories of each topic.

```
Topic 0:
Family & Parenting    932
Love & Dating         626
Health & Safety       548

Topic 1:
Etiquette & Ethics     829
Family & Parenting     416
Friends & Neighbors    305

Topic 2:
Family & Parenting    1382
Marriage & Divorce     933
Love & Dating          546
```

Just like KMeans and NMF, it seems to be doing well.

Furthermore there's a handy tool called [LDAvis](https://github.com/cpsievert/LDAvis) that allows us to visualize the clustering. It does so by dimensional scaling to project the topics onto 2D space.

We see here each topic and their relevant keywords.
- On the left, the area of the circle refers to the prevelance of the topics. 
- On the right, the horizontal bar refers the keywords most useful in interpreting the topic. In addition, we see how relevant they are similar to feature importance. 

![lda vis](/image/ldavis.png)

The [html applet](https://htmlpreview.github.io/?https://github.com/unit-00/dearabby/blob/master/model/lda.html) is here to be played with.

## Evaluation
For a baseline evaluation, since this is an unsupervised learning project we do not have an objective metrics.  

Based on the average length of the categories, a human would guess there are 2 or 3 large categories.

Through visualization and interpretation, we see the models being able to categorize similarly with more details such as the particular topics and the keywords associated with them.


## Application
Through topic modelling, we are able to extract issues that people are writing in about. In a business context, we can use this to detect people's concerns. With a time component, we might be able to see how concerns evolve over time. 

A more subtle issue I was able to deal with here is multiclass-multilabel. Each articles has multiple categories and by topic modelling we're able to group them for a cleaner classification purpose. As we do not have a clean metrics for multiclass-multilabel, to my knowledge. However this does also come with the conerns of drowning out more hidden issues people may write in.


## Conclusion
- People are writing in about family, relationship, love, and etiquette issues.
- Unsupervised learning is powerful and is able help us understand text topics and along with categorizing them.
- Fallbacks
    - No objective metrics to determine whether we're doing a good job
    - Requires domain knowledge in interpreting the models' performance.

## Follow up
Couple ideas I have regarding the next steps are
- Text generation for answers
- Text summarizing the questions
- Autolabel each articles

## Credit
- Text mined belong to Dear Abby 
- Thank you g119 for the memories
- And Tom for introducing me to pyLDAvis

## Reproduce model
I have added the conda environment for the tools used and the data collected. 

You can create a conda environment with `conda env create --file dearabby-env.txt`

This will provide all the tools used for the project.

If you would like to scrape the data, be sure to set up a mongo environment. I used docker to setup a mongo server for a warehouse.

`docker run --name mongoserver -p 27017:27017 -d mongo`

After starting the mongo server, you can run `python data/collect_data.py` to scrape the data and have it inserted into your mongo server.

However I have included the data I have collected myself inside the data directory as well.

In which case, you can run `python model/run_models.py` and have it print out the results of the three models. 

If you would like to see the visualizations, I have included the notebooks as well.
