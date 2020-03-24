#  Evaluation Metrics for Recommenders

1. Rating Metrics: These are used to evaluate how accurate a recommender is at predicting ratings that users gave to items

   >**Input:**
   >	rating_true-- True rating data. There should be no duplicate *(userID, itemID, $r_{i}$)* pairs
   >	rating_pred-- Predicted rating data. There should be no duplicate (userID, itemID, $\hat{r}_{i}$) pairs

   - Root Mean Square Error (**RMSE**) - measure of average error in predicted ratings
     $$
     \mathrm{RMSE}=\sqrt{\frac{\sum_{i=1}^{n}\left(\hat{r}_{i}-r_{i}\right)^{2}}{n}} 
     \tag{1}
     $$

   - Mean Absolute Error (**MAE**) - similar to RMSE but uses absolute value instead of squaring and taking the root of the average
     $$
     \mathrm{MAE}=\frac{\sum_{i=1}^{n}\left|\hat{r}_{i}-r_{i}\right|}{n}
     \tag{2}
     $$

   - R Squared (**R2**) - essentially how much of the total variation is explained by the model
     $$
     R^2 = 1 - \frac{\sum_{i=1}^{n} (r_i - \hat{r}_i)^2}{\sum_{i=1}^{n} (r_i - \bar{r})^2}
     \tag{3}
     $$
     where $\bar{r} =  \frac{1}{n} \sum_{i=1}^{n } r_i$ is the mean of the real scores.

   - Explained Variance - how much of the variance in the data is explained by the model
     $$
     \text {explained-var}=1-\frac{\operatorname{Var}\{r-\hat{r}\}}{\operatorname{Var}\{r\}}
     \tag{4}
     $$
     where $Var\{\cdot\}$ is the variance of a list.
     
     | Metric             | Range    | Selection criteria            | Limitation                                   | Reference                                                    |
     | ------------------ | -------- | ----------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
     | RMSE               | $> 0$    | The smaller the better.       | May be biased, and less explainable than MSE | [link](https://en.wikipedia.org/wiki/Root-mean-square_deviation) |
     | R2                 | $\leq 1$ | The closer to $1$ the better. | Depend on variable distributions.            | [link](https://en.wikipedia.org/wiki/Coefficient_of_determination) |
     | MAE                | $\geq 0$ | The smaller the better.       | Dependent on variable scale.                 | [link](https://en.wikipedia.org/wiki/Mean_absolute_error)    |
     | Explained variance | $\leq 1$ | The closer to $1$ the better. | Depend on variable distributions.            | [link](https://en.wikipedia.org/wiki/Explained_variation)    |

2. Ranking Metrics: These are used to evaluate how relevant recommendations are for users

   - Precision - this measures the proportion of recommended items that are relevant
$$
\text { precision }=\frac{ |\{\text { true items}\} \cap\{\text { predicted items }\}| }{ |\{\text { predicted items }\} |}\tag{5}
$$
   - Recall - this measures the proportion of relevant items that are recommended
$$
\text { recall }=\frac{ |\{\text { true items}\} \cap\{\text { predicted items }\}| }{ |\{\text { true items }\} |}\tag{6}
$$
   - F-score - The weighted harmonic mean of precision and recall, the traditional F-measure or balanced F-score is:

$$
F=\frac{2 \cdot \text { precision } \cdot \text { recall }}{(\text { precision }+\text { recall })}\tag{7}
$$

   - Normalized Discounted Cumulative Gain (NDCG) - evaluates how well the predicted items for a user are ranked based on relevance
$$
   \mathrm{DCG}_{\mathrm{p}}=\sum_{i=1}^{p} \frac{r e l_{i}}{\log _{2}(i+1)}\\
   \mathrm{nDCG}_{\mathrm{p}}=\frac{D C G_{p}}{I D C G p}\tag{8}
$$

   - Mean Average Precision (MAP) - average precision for each user normalized over all users
$$
   \mathrm{MAP}=\frac{\sum_{q=1}^{Q} \mathrm{AveP}(\mathrm{q})}{Q}\tag{9}
$$

   - Arear Under Curver (AUC) - integral area under the receiver operating characteristic curve

   >When calculate the logloss, the rating should be binary and prediction should be float number ranging
   >  from 0 to 1.

   

   - Logistic loss (Logloss) - the negative log-likelihood of the true labels given the predictions of a classifier 
     $$
     L_{\log }=-\log \operatorname{Pr}(r | \hat{r})=-\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K}(r \log (\hat{r})+(1-r) \log (1-\hat{r}))\tag{10}
     $$

     > When calculate the logloss, the rating should be binary and prediction should be float number ranging
     >   from 0 to 1.

| Metric    | Range                 | Selection criteria                                           | Limitation                                                   | Reference                                                    |
| --------- | --------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Precision | $\geq 0$ and $\leq 1$ | The closer to $1$ the better.                                | Only for hits in recommendations.                            | [link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems) |
| Recall    | $\geq 0$ and $\leq 1$ | The closer to $1$ the better.                                | Only for hits in the ground truth.                           | [link](https://en.wikipedia.org/wiki/Precision_and_recall)   |
| NDCG      | $\geq 0$ and $\leq 1$ | The closer to $1$ the better.                                | Does not penalize for bad/missing items, and does not perform for several equally good items. | [link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems) |
| MAP       | $\geq 0$ and $\leq 1$ | The closer to $1$ the better.                                | Depend on variable distributions.                            | [link](https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems) |
| AUC       | $\geq 0$ and $\leq 1$ | The closer to $1$ the better. 0.5 indicates an uninformative classifier | Depend on the number of recommended items (k).               | [link](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) |
| Logloss   | $0$ to $\infty$       | The closer to $0$ the better.                                | Logloss can be sensitive to imbalanced datasets.             | [link](https://en.wikipedia.org/wiki/Cross_entropy#Relation_to_log-likelihood) |

