This is a feature selection tool that can be used to select features for machine learning tasks.<br>

Most ideas of this tool are from scikit-learn.<br>

Feature selection is a a process of finding good features for machine learning models. Feature selection is known as an important part of feature engineering.
We can not always use all features to build a machine learning model. Datasets can somtimes contain over 1000 features, selecting the best features is a crucial step to build a model which performaces well on the test set.<br>
Feature selection can also be considered as a kind of dimensionality reduction mehod. That means we will have smaller dataset, faster training.<br>

Feature generation is another topic of feature engineering. Sometimes two feature are highly correlated, merge this two features mathematically could create a much more meaningful feature for a ML model. Feature generation is a large topic and will not be disscused here.<br>

This feature selection tool provides the following useful methods:

    1. select_high_variance_features
           find all features whose variance meet the threshold.
    2. select_features_by_feature_scores
           find all feature by feature scores which are calculated by a score function.
    3. get_feature_coefficients
           get feature coefficients by fitting a linear model. norm prior can be specified to
           identify more or less important features.
    4. get_feature_importances
           get feature importance using xgboost with cross-validation. This is a different
           approach from a linear model.
    5. get_feature_ranking_by_recursively_eliminating_features
           A recursive feature elimination method to find the best n features.
    6. get_new_features_by_clustering_features
           create new features by clustering.
           This is not a feature selection method.
