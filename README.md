# Personalize-Expedia-Hotel-Searches
 Data Mining assignment 2

##Data downloaded from here (unzip data.zip):
https://www.kaggle.com/competitions/expedia-personalized-sort/data


##Sources

Code for 4th best score (VU students 3 years ago):
https://github.com/igorpejic/personalize_expedia_hotel_searches_2013

Paper from 5th best submission:
https://www.arxiv-vanity.com/papers/1311.7679/


Not so good submission but nice summary:
http://www.davidwind.dk/wp-content/uploads/2014/07/main.pdf


Final presentation from the competition + dataset description:
https://www.dropbox.com/sh/5kedakjizgrog0y/_LE_DFCA7J/ICDM_2013


Discussion on Kaggle:
https://www.kaggle.com/competitions/expedia-personalized-sort/discussion/6228

TODO:
1. Outlier detection: look for outliers on city and country level when replacing them with mean per category
2. Ranking baseline
3. Missing values. Catboost does a weird thing
4. Oleg: Feature pruning and importance with SHAP https://catboost.ai/en/docs/features/feature-importances-calculation
https://colab.research.google.com/github/catboost/tutorials/blob/master/feature_selection/select_features_tutorial.ipynb#scrollTo=hCEUEOb_SqEk
4. Oleg: Try classification once again
   1. Random forest
   2. Extreme trees
5. Try lambdaMART (+xgboost\lgbm optimising lambdaMART) https://github.com/sophwats/XGBoost-lambdaMART/blob/master/LambdaMART%20from%20XGBoost.ipynb
6. Add negative sampling for non-matched pairs
7. Add random baseline and evaluate internally
8. Add baseline from https://github.com/benhamner/ExpediaPersonalizedSortCompetition to our baselines
9. Boosting: do not encode site_id, prop_id etc - they have to be naturally granular
10. add ordinal categories to catboost
11. dates of the staying + its features. add holidays
    1. business trip = short and workday/non-weekend
    2. close to holiday +-3 days
    3. is a day off during a week day
    4. add days of week for start for example
12. aggregations for:
    4. people
    5. months\day\season\weekday (sales per time period)
13. calculate avg tax per country df['usr_extra_pay'] = df['gross_bookings_usd'] - df['price_usd']
14. add the difference between stars visitor_hist_starrating, prop_starrating, prop_review_score. also normalize by the price
15. adjust the star by one if it's a chain
16. prop_location_score1 and 'prop_location_score2' may be correlated to the duration of stay
17. Agoston: find the normalized price: check if the price for the same hotel is really different (mb for different countries of number of days)
18. Agoston: Correlation between adv and position
19. Agoston: compare date and distribs betweeen train/test
20. Agoston: hists for comp1_rate
21. convert absolute percentage difference with competitor to the money difference
22. Agoston: comp_rate is 0 -> comp_rate_perc_diff should be 0. But it has a value. Any idea?
23. Having aggregations, try the difference between the current month and the prev, for example
24. Climate/weather in the src and dst places + difference?
25. hotel_cumulative_share, a measure of how often a hotel has been booked previously, and previous_user_hotel_interaction, a categorical variable indicating if a user had clicked or purchased this hotel previously, are the top 2 most important features for our logged-in users. Coalescing a hotel’s purchase history into learned “embeddings” using latent factor models may add significant value to the model.
26. Add SVD-based recsys
27. Add default model
28. Train model on train+val combined
29. use position as a feature but ONLY when random is False
30. copy stats for position to the subm df directly
31. prop_review_score - 0 means there have been no reviews, null that the information is not available. What to do?
32. Add memory cleaning from kaggle to the main notebook
33. Catboost split evaluation into batches and avg


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
8. 


# Tried, not worked:
1. CatBoostRanker with ```'eval_metric': 'NDCG:top=5;type=Base;denominator=LogPosition'``` continued training after optimum ```'loss_function': 'QueryRMSE'``` and overfitted. The score from LB correlates right with the place when we started overfitting 
