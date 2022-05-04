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
1. Add negative sampling for non-matched pairs
2. Add random baseline and evaluate internally
3. Add baseline from https://github.com/benhamner/ExpediaPersonalizedSortCompetition to our baselines
4. Boosting: do not encode site_id, prop_id etc - they have to be naturally granular
5. Add eval func for ranking and classification 
6. make optimizition oriented on ndcg regardless of the task (ranking, clf)
7. add ordinal categories ot catboost
8. Calculate the dates of the staying + its features using the date of booking + the days shift
9. 
```
df["hour"] = df["date_time"].dt.hour
df["minute"] = df["date_time"].dt.minute
df["early_night"] = ((df["hour"]>19) | (df["hour"]<3)) # no added value from feature
```
11. switch to parquet from csv

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
2. 

