# Personalize Expedia Hotel Searches - ICDM 2013
Data Mining assignment 2, group 65  
Oleg Litvinov  
Agoston Szabo  
Nedim Azar

## Original competition

https://www.kaggle.com/competitions/expedia-personalized-sort/overview

## Data downloaded from here:
https://www.kaggle.com/competitions/2nd-assignment-dmt2022/data


## Sources

* Code for 4th best score (VU students 3 years ago):
https://github.com/igorpejic/personalize_expedia_hotel_searches_2013
* Paper from 5th best submission:
https://www.arxiv-vanity.com/papers/1311.7679/
* Not so good submission but nice summary:
http://www.davidwind.dk/wp-content/uploads/2014/07/main.pdf
* Final presentation from the competition + dataset description:
https://www.dropbox.com/sh/5kedakjizgrog0y/_LE_DFCA7J/ICDM_2013
* Discussion on Kaggle:
https://www.kaggle.com/competitions/expedia-personalized-sort/discussion/6228

# TODOs

## EDA
1. prop_location_score1 and 'prop_location_score2' may be correlated to the duration of stay
2. Correlation between adv and position

## Preprocessing
1. Preprocess data for train_val together
2. Missing values. Catboost does only min or max
3. Outliers
   1. Outlier detection: look for outliers on city and country level when replacing them with mean per category 
   2. Price anomaly detection https://www.kaggle.com/code/nikitsoftweb/production-time-series-of-price-anomaly-detection/notebook
4. Add negative sampling for non-matched pairs
5. Return shuffle split back to refresh the distrib from time to time
6. Fill missing values with some historicals, competitors? 
7. Fill missing prop_review_score, prop_location_score2, srch_query_affinity_score values with the worst case scenario?
8. convert absolute percentage difference with competitor to the money difference (The absolute percentage difference 
(if one exists) between Expedia and competitor N’s price (Expedia’s price the denominator))
9. Price_diff and star ranks

## Features
1. add visitor_hist_adr_usd and 'prop_historical_price' to comparison_col when build features for price
2. dates of the staying + its features.
   1. business trip = short and workday/non-weekend
   2. close to holiday +-3 days
   3. is a day off during a week day 
   4. add boolean for a weekend
3. aggregations for day\weekday (sales per time period)
   1. Having aggregations, try the difference between the current month and the prev, for example
4. calculate avg tax per country df['usr_extra_pay'] = df['gross_bookings_usd'] - df['price_usd']
5. order of the hotel 
    1. for this month
    2. for this dst region
    3. from this search region
    4. for this booking period
6. hotel_cumulative_share, a measure of how often a hotel has been booked previously, and 
previous_user_hotel_interaction (how), a categorical variable indicating if a user had clicked or purchased this hotel 
previously, are the top 2 most important features for our logged-in users. Coalescing a hotel’s purchase history into 
learned “embeddings” using latent factor models may add significant value to the model.
7. Do date_time_ordinal for date, not datetime?
8. order by previous booking prob
9. Add std for groups

## Modeling
1. Baselines
   2. Ranking baseline
   3. Add default model (for each dst country predict the most famous hotels)
   4. Add baseline from https://github.com/benhamner/ExpediaPersonalizedSortCompetition to our baselines 
2. Try lambdaMART (+xgboost\lgbm optimising lambdaMART) https://github.com/sophwats/XGBoost-lambdaMART/blob/master/LambdaMART%20from%20XGBoost.ipynb
3. Boosting: do not encode site_id, prop_id etc - they have to be naturally granular
4. add ordinal categories to catboost
5. try classification once again
      1. Random forest
      2.Extreme trees 
6. Catboost split evaluation into batches and avg in case of memory issues

## Unsorted backlog
1. use position as a feature but ONLY when random is False 
2. copy stats for position to the subm df directly
3. prop_review_score - 0 means there have been no reviews, null that the information is not available. What to do?
4. per hotel: booking/counting, click/counting



# DONE:
1. Normalize price per number of nights
2. aggregations for:
    1. visitor country
    2. destination country 
    3. hotel (property id)
    4. search id
3. business trip = short, no children, 1 adult
4. prop_location_score1 and 'prop_location_score2' may be correlated to the duration of stay: no strong correlation
5. Calculate the dates of the staying + its features using the date of booking + the days shift. add holidays
6. Order the price within the srch_id, srch_destination_id, prop_id
7. Calculate mean target values per propert_id


# Storytelling:
1. 02.05.2022
```
CatBoostRanker(iterations=500, 
               loss_function='QueryRMSE'
               early_stopping_rounds=50)
cat_cols = ['site_id', 'visitor_location_country_id',
            'prop_country_id', 'prop_id', 'srch_destination_id']
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.1, shuffle=False)  # doesn't preserve groups
```
Public LB: 0.32661
2. 04.05.2022
Added feature engineering and leaked with the price with tax. The score anyway improved
Public LB: 0.36825
3. 05.05.2022 
   1. added features only
   Public LB: 0.36995 
   2. submission 4 with new features shew overfit with local 0.36886 and public 0.36163
4. YetiRank is added http://proceedings.mlr.press/v14/gulin11a.html 
Validation is changed into GroupShuffleSplit
Public LB: 0.33767
5. Simple validation split is added. srch_id % 10 == 5 -> test, srch_id % 10 == 1 -> val, rest ->train. Idea from https://arxiv.org/pdf/1311.7679v1.pdf 
Only 200 epochs.  
Train: 0.07871  
Val: 0.35554  
Test: 0.35682   
Public LB: 0.31550
6. 08.05.2022
Public LB: 0.33190
7. 09.05.2022
Fitted for 2.5k iterations on Kaggle
Public LB: 0.38823
8. 10.05.2022, 3.75k iterations (early stop), original + prop location + historical price, Public LB: 0.39121
9. 11.05.2022
11th submission. CatBoostRanker, bestIteration = 4991. 157 features without aggregations (only price order per srch and dst ids)  
Train: 0.45396  
Val: 0.39305  
Test: 0.39381  
Public LB: 0.39492  
12th submission. CatBoostRanker, bestIteration = 4997. 198 added new aggregated features  
Val: 0.40018  
Public LB: 0.40190
10. 14.05.2022
13th submission. SVD
Public: 0.27837
11. 15.05.2022
14th submissio. LGBM
Public: 0.38713
12. 16.05.2022 
Did a long selection of about a third (others were even not tried) aggregation parameters (some were checked with LossValueChange while the others with SHAP)  
Train: 0.4586  
Val: 0.39704  
Test: 0.39461  
Public LB: 0.40357  
Trained for 5.5k epochs. The validation started distorting
13. 17.05.2022
16th submission.
Switched to YetiRankPairwise. Trained twice faster and with better results. Worked a lot on feature selection. Processed
about a half. Removed ones with 0 shap values.  
Train: 0.44847  
Val: 0.40284  
Test: 0.39971  
Public LB: 0.40694  
17th submission.
Trained on train + val, stopped on test. NOT ALLOWED TO ANALYSE, JUST TO CHECK THE BOOST FROM THE TRICK  
Public LB: 0.40920
14. 19.05.2022
18th submission
LGBM trained on a lot of predictors
Public score: 0.39566
15. 24.05.2022
19th submission
Trained on 400 features. 8000 epochs. Fair metrics:  
"train_NDCG@5": 0.45839115000448594,  
"val_NDCG@5": 0.40318595229960297,  
"test_NDCG@5": 0.39889310524098376  
Retrained on train+val. Public score: 0.40845
16. 24.05.2022
20th submission
Blending of 18th and 19th submissions. weight 0.65 for catboost and 0.35 for lgbm.  
Public score: 0.40878
17. 26.05.2022
21th submission
Large hyper-parameter tuning. Generally, increased LR. Tuned on train vs val. Retrained on train+val using 226 selected
features.  
"train_NDCG@5": 0.49272081873197615,  
"val_NDCG@5": 0.4040699997013831,  
"test_NDCG@5": 0.40020674965410236  
Public score: 0.41120  
18. 27.05.2022
22th submission
248 features, default parameters without refit
"train_NDCG@5": 0.4537878607462358,  
"val_NDCG@5": 0.4080801244174121,  
"test_NDCG@5": 0.4049248757109859  
Public score: 0.41247  
**The competition is ended**
19. 28.05.2022
23th submission. 189 features, no downsampling. 10 rounds of skopt.  
`{'learning_rate': 0.18955396053267756, 
'loss_function': 'YetiRank', 
'nan_mode': 'Min', 
'l2_leaf_reg': 0.7119418600172993}`  
"train_NDCG@5": 0.47907378294452035,  
"val_NDCG@5": 0.40869611361177177,  
"test_NDCG@5": 0.4046275150950173  
Public score: 0.41418  



# Open questions
1. comp_rate is 0 -> comp_rate_perc_diff should be 0. But it has a value.

# Tried, not worked:
1. CatBoostRanker with ```'eval_metric': 'NDCG:top=5;type=Base;denominator=LogPosition'``` continued training after optimum ```'loss_function': 'QueryRMSE'``` and overfitted. The score from LB correlates right with the place when we started overfitting 
