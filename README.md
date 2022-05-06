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
1. Oleg: Try classification once again
2. Oleg: check losses close to NDCG as there is a mismatch during optimization
3. Oleg: add evaluation on train
4. Oleg: try simplified CV and compare with LB
5. Add negative sampling for non-matched pairs
6. Add random baseline and evaluate internally
7. Add baseline from https://github.com/benhamner/ExpediaPersonalizedSortCompetition to our baselines
8. Boosting: do not encode site_id, prop_id etc - they have to be naturally granular
9. add ordinal categories ot catboost
10. Calculate the dates of the staying + its features using the date of booking + the days shift. add holidays
    1. business trip = short and workday/non-weekend
    2. close to holiday +-3 days
    3. is a day off during a week day
11. Feature prooning and importance with SHAP
12. average country price
13. aggregations for:
    1. countries (stars per country avg)
    2. location
    3. hotels
    4. people
    5. months\day\season\weekday (sales per time period)
14. calculate avg tax per country df['usr_extra_pay'] = df['gross_bookings_usd'] - df['price_usd']
15. add the difference between stars visitor_hist_starrating, prop_starrating, prop_review_score. also normalize by the price
16. adjust the star by one if it's a chain
17. prop_location_score1 and 'prop_location_score2' may be correlated to the duration of stay
18. Agoston: find the normalized price: check if the price for the same hotel is really different (mb for different countries of number of days)
19. Normalize price per number of nights
20. Order the price within the srch_id, destination loc_id
21. Agoston: Correlation between adv and position
22. Agoston: compare date and distribs betweeen train/test
23. optional. Agoston: unsupervised private\business trip (children, ..)
24. Agoston: hists for comp1_rate
25. convert absolute percentage difference with competitor to the money difference
26. Agoston: comp_rate is 0 -> comp_rate_perc_diff should be 0. But it has a value. Any idea?
27. Having aggregations, try the difference between the current month and the prev, for example

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
3. 


# Tried, not worked:
1. CatBoostRanker with ```'eval_metric': 'NDCG:top=5;type=Base;denominator=LogPosition'``` continued training after optimum ```'loss_function': 'QueryRMSE'``` and overfitted. The score from LB correlates right with the place when we started overfitting 
