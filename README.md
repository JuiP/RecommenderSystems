Recommender System
--------------------------------------------------------------------------------------------------
***CS F469 IR Assignment - 3***

**Problem Statement**:

The task is to compare various techniques used in implementing Recommender Systems on the basis of their errors using Root Mean Square Error, Precision on top K and Spearman Rank Correlation. Also compare their overall running time and prediction time.
The techniques implemented and compared are:
1. Collaborative Filtering.
2. Collaborative Filtering using Baseline approach.
3. Singular Value Decomposition(SVD).
4. SVD with 90% energy.
5. CUR.
6. CUR with 90% energy.

**About the project**

*Dataset used -*
[Movie lens Dataset](https://grouplens.org/datasets/movielens/) is used in this assignment consisting of 6040 users rating of 3883 movies

Have a look at the file [Design Document](https://github.com/JuiP/RecommenderSystems/blob/master/Design%20Document.pdf). It includes the concepts used along with the time taken for each implementation step.

Project By:
- **Kriti Jethlia**: Email- <f20180223@hyderabad.bits-pilani.ac.in>
- **Jui Pradhan**: Email- <f20180984@hyderabad.bits-pilani.ac.in>
- **Anusha Agarwal**: Email- <f20180032@hyderabad.bits-pilani.ac.in>
--------------------------------------------------------------------------------------------------
**How to run the code**
--------------------------------------------------------------------------------------------------

1. Clone the repository : https://github.com/JuiP/RecommenderSystem.git
2. cd RecommenderSystem
3. Change the path to ratings.dat in the python code, wherever you have saved the ml-1m folder(preferably in the same folder) and run the python script for each method.
4. Run file: 

              python3 collab_user.py
              python3 collab_global.py
              python3 cur.py
              python3 cur_90.py
              python3 svd.py
              python3 svd_90.py
              
  
5. The Root Mean Square Error(RMSE), Precision on top K, Spearman Rank Correlation and Time taken for prediction is printed at the end of each file execution.:)

---------------------------------------------------------------------------------------------------
**Dependencies/modules used**
---------------------------------------------------------------------------------------------------
- time
- math
- pandas
- Numpy
- random
