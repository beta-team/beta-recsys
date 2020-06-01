# DataSet Tutorial For BETA-Recsys

**Author: **Yucheng Liang

**Last Update Time: **2020-05-29

## Introduction

BETA-Recsys provides users a wide range of datasets for recommendation system training. For convenience, we preprocess a number of datasets for you to train, getting you rid of splitting them on you local machine. Also this framework provides users a set of useful interfaces for data split.

---

## Dataset Statistics

Here we present some basic staticstics for the datasets in our framework.

|                         **Dataset**                          | **Interactions** | **Baskets** | **Temporal** |
| :----------------------------------------------------------: | :--------------: | :---------: | :----------: |
| [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) |        ✔️         |      ✖️      |      ✔️       |
| [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) |        ✔️         |      ✖️      |      ✔️       |
| [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) |        ✔️         |      ✖️      |      ✔️       |
|    [Last.FM](https://grouplens.org/datasets/hetrec-2011/)    |        ✔️         |      ✖️      |      ✖️       |
| [Epinions](http://www.trustlet.org/downloaded_epinions.html) |        ✔️         |      ✖️      |      ✖️       |
| [Tafeng](https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/) |        ✔️         |      ✖️      |      ✔️       |
| [Dunnhumby](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey) |        ✔️         |      ✔️      |      ✔️       |
| [Instacart](https://www.instacart.com/datasets/grocery-shopping-2017) |        ✔️         |      ✖️      |      ✔️       |
|    [citeulike-a](https://github.com/js05212/citeulike-a)     |        ✔️         |      ✖️      |      ✖️       |
| [citeulike-t](https://github.com/changun/CollMetric/tree/master/citeulike-t) |        ✔️         |      ✖️      |      ✖️       |
|     [HetRec](http://ir.ii.uam.es/hetrec2011/) MoiveLens      |        ✔️         |      ✖️      |      ✔️       |
|     [HetRec](http://ir.ii.uam.es/hetrec2011/) Delicious      |        ✔️         |      ✔️      |      ✖️       |
|       [HetRec](http://ir.ii.uam.es/hetrec2011/) LastFM       |        ✔️         |      ✔️      |      ✔️       |
|             [Yelp](https://www.yelp.com/dataset)             |        ✔️         |      ✖️      |      ✔️       |
|  [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)  |        ✔️         |      ✖️      |      ✔️       |
| [Yoochoose](https://2015.recsyschallenge.com/challenge.html) |                  |             |              |
|    [Diginetica](https://cikm2016.cs.iupui.edu/cikm-cup/)     |                  |             |              |
| [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) |                  |             |              |
| [Ali-mobile](https://tianchi.aliyun.com/dataset/dataDetail?dataId=46) |                  |             |              |
| [Retailrocket](https://www.kaggle.com/retailrocket/ecommerce-dataset#events.csv) |                  |             |              |

Because some split methods require a specific features, like `random_basket` expect the dataset has a **Basket** column. Here we list all the split methods for each dataset.

The prerequisite for each split methods are:

+ `leave_one_out`: none
+ `leave_one_basket`: require a **Basket** column in dataset
+ `random`: none
+ `random_basket`: require a **Basket** column in dataset
+ `temporal`: require a **Timestamp(Temporal)** column in dataset
+ `temporal_basket`: require a **Timestamp(Temporal)** and a **Basket** column in dataset

|                         **Dataset**                          | **leave_one_out** | **leave_one_basket** | **random** | **random_basket** | **temporal** | temporal_basket |
| :----------------------------------------------------------: | :---------------: | :------------------: | :--------: | :---------------: | :----------: | :-------------: |
| [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✔️       |        ✖️        |
| [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✔️       |        ✖️        |
| [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) |         ✔️         |          ✖️           |     ✔️      |                   |              |                 |
|    [Last.FM](https://grouplens.org/datasets/hetrec-2011/)    |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✖️       |        ✖️        |
| [Epinions](http://www.trustlet.org/downloaded_epinions.html) |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✖️       |        ✖️        |
| [Tafeng](https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/) |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✔️       |        ✖️        |
| [Dunnhumby](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey) |         ✔️         |          ✔️           |     ✔️      |         ✔️         |      ✔️       |        ✔️        |
| [Instacart](https://www.instacart.com/datasets/grocery-shopping-2017) |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✔️       |        ✖️        |
|    [citeulike-a](https://github.com/js05212/citeulike-a)     |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✖️       |        ✖️        |
| [citeulike-t](https://github.com/changun/CollMetric/tree/master/citeulike-t) |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✖️       |        ✖️        |
|     [HetRec](http://ir.ii.uam.es/hetrec2011/) MoiveLens      |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✔️       |        ✖️        |
|     [HetRec](http://ir.ii.uam.es/hetrec2011/) Delicious      |         ✔️         |          ✔️           |     ✔️      |         ✖️         |      ✖️       |        ✖️        |
|       [HetRec](http://ir.ii.uam.es/hetrec2011/) LastFM       |         ✔️         |          ✔️           |     ✔️      |         ✔️         |      ✔️       |        ✔️        |
|             [Yelp](https://www.yelp.com/dataset)             |         ✔️         |          ✖️           |     ✔️      |                   |              |                 |
|  [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)  |         ✔️         |          ✖️           |     ✔️      |         ✖️         |      ✖️       |        ✖️        |
| [Yoochoose](https://2015.recsyschallenge.com/challenge.html) |         ✔️         |                      |     ✔️      |                   |              |                 |
|    [Diginetica](https://cikm2016.cs.iupui.edu/cikm-cup/)     |         ✔️         |                      |     ✔️      |                   |              |                 |
| [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) |         ✔️         |                      |     ✔️      |                   |              |                 |
| [Ali-mobile](https://tianchi.aliyun.com/dataset/dataDetail?dataId=46) |         ✔️         |                      |     ✔️      |                   |              |                 |
| [Retailrocket](https://www.kaggle.com/retailrocket/ecommerce-dataset#events.csv) |         ✔️         |                      |     ✔️      |                   |              |                 |

Also, we provide some information about the dataset content such as the number of items, users and so on. This may give you a brief view of the dataset.

|                         **Dataset**                          |    Rows    |   User    | **Item**  | Rating | **Timestamp** |
| :----------------------------------------------------------: | :--------: | :-------: | :-------: | :----: | :-----------: |
| [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) |  100,000   |    943    |   1,682   |   5    |    49,282     |
| [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) | 1,000,209  |   6,040   |   3,706   |   5    |    458,455    |
| [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) | 25,000,095 |  162,541  |  59,047   |   10   |  20,115,267   |
|    [Last.FM](https://grouplens.org/datasets/hetrec-2011/)    |   92,834   |   1,892   |  17,632   | 5,436  |       1       |
| [Epinions](http://www.trustlet.org/downloaded_epinions.html) |  664,825   |  40,163   |  139,738  |   5    |       1       |
| [Tafeng](https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset/) |            |           |           |        |               |
| [Dunnhumby](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey) |    500     |    28     |    463    |   1    |      N/A      |
| [Instacart](https://www.instacart.com/datasets/grocery-shopping-2017) | 33,819,106 |  206,209  |  49,685   |   1    |   3,346,083   |
|    [citeulike-a](https://github.com/js05212/citeulike-a)     |  204,986   |    240    |  16,980   |   1    |       1       |
| [citeulike-t](https://github.com/changun/CollMetric/tree/master/citeulike-t) |  134,860   |    216    |  25,584   |   1    |       1       |
|     [HetRec](http://ir.ii.uam.es/hetrec2011/) MoiveLens      |  855,598   |   2,113   |  10,109   |   10   |    809,328    |
|     [HetRec](http://ir.ii.uam.es/hetrec2011/) Delicious      |  437,593   |   1,867   |  69,223   |   1    |    104,093    |
|       [HetRec](http://ir.ii.uam.es/hetrec2011/) LastFM       |  186,479   |   1,892   |  12,523   |   1    |     9,749     |
|             [Yelp](https://www.yelp.com/dataset)             | 8,021,122  | 1,968,703 |  209,393  |   5    |   7,853,102   |
|  [Gowalla](https://snap.stanford.edu/data/loc-Gowalla.html)  | 6,442,892  |  107,092  | 1,280,969 |   1    |   5,561,957   |
| [Yoochoose](https://2015.recsyschallenge.com/challenge.html) | 1,150,753  |  509,696  |    735    |   1    |    19,949     |
|    [Diginetica](https://cikm2016.cs.iupui.edu/cikm-cup/)     | 1,235,380  |  310,324  |  122,993  |   1    |      152      |
| [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) | 3,835,331  |  37,376   |  930,607  |   1    |    698,889    |
| [Ali-mobile](https://tianchi.aliyun.com/dataset/dataDetail?dataId=46) | 12,256,906 |  10,000   | 2,876,947 |   1    |       1       |
| [Retailrocket](https://www.kaggle.com/retailrocket/ecommerce-dataset#events.csv) | 2,756,101  | 1,407,58  |  235,061  |   1    |   2,749,921   |

---

## Dataset Usage

### Download Data

BETA-Recsys provides download interface for users to download different dataset. Here is an example:

```python
import sys
import os
sys.path.append(os.path.abspath('.'))
from beta_rec.datasets.movielens import Movielens_1m

movielens_1m = Movielens_1m()
movielens_1m.download()
```

However, not every dataset could be downloaded directly with our framework. **For some datasets, you will still have to download them manually. You are supposed to follow our tips to download and put the dataset in the correct folder in order to be detected by our framework.** 

### Load Data

Downloading and preprocessing giant datasets may be a disturbing things, and in order to deal with this issue, we have preprocessed a wide range of datasets and stored the processed data in our remote server. Users can access them easily by using our `load` function.

```python
import sys
import os
sys.path.append(os.path.abspath('.'))
from beta_rec.datasets.movielens import Movielens_1m

movielens_1m = Movielens_1m()
movielens_1m.load_leave_one_out()
movielens_1m.load_random_split()
```

Due to storage limitation, we only store a copy of split data with default parameters. If you want a custom split, you'll still have to split them on you local machine.

### Make Data

Users can simply ignore these functions because when you use custom parameters in `load` functions, it will automatically call `make` functions. So you don't need to care about this functions. **We strongly recommend you to use `load` function directly in most of you time.**

---

## Data Split

For users who are willing to split some datasets that is not covered by our framework, we still provide various methods to make it easy to split huge data, without caring the details. There are 6 main methods for users to split data.

### random_split

This method will select a portion of records based on the given `test_rate` randomly.

### random_basket_split

This method will select a portion of baskets(one basket may cover more than one record) based on the given `test_rate` randomly.

### leave_one_out

This method will first rank all the records by time (if a timestamp column is provided), and then select the last record.

### leave_one_basket

This method will first rank all the records by time (if a timestamp column is provided), and then select the last basket.

### temporal_split

This method will first rank all the records by time (if a timestamp column is provided), and then select the last portion of records.

### temporal_basket_split

This method will first rank all the records by time (if a timestamp column is provided), and then select the last portion of baskets.

## More

For any quesitons, please tell us by **creating an issue** or contact us by sending an email to recsys.beta@gmail.com. We will try to respond it as soon as possible.