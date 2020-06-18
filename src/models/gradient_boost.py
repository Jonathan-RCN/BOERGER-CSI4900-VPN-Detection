import src.config as cfg
import src.feature_subsets as subset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime


def main():
    connection_rw_size = cfg.CONNECTION_RW_SIZE
    timw_rw_size_min = cfg.TIME_RW_SIZE

    features_csv = f'../../data/processed/full_ft_netflow_crw_{connection_rw_size}_trw_{timw_rw_size_min}_2.csv'
    label=subset.get_target(features_csv)

    baseline_ft, baselone_ft_name=subset.get_baseline_ft(features_csv)
    time_ft, time_ft_name=subset.get_time_ft(features_csv)
    conn_ft, conn_ft_name=subset.get_conn_ft(features_csv)
    comb_ft, comb_ft_name=subset.get_comb_ft(features_csv)

    baseline_predictions, baseline_y_test, baseline_model= gradient_boost_classifier(baseline_ft, label, random_state=42)
    evaluate_ml_model_results(baseline_predictions,baseline_y_test,baseline_model,baseline_ft,label)


    print('/////////////////////////')
    time_predictions, time_y_test, time_model=gradient_boost_classifier(time_ft, label, random_state=42)
    evaluate_ml_model_results(time_predictions,time_y_test, time_model, time_ft, label)

    print('/////////////////////////')

    conn_predictions, conn_y_test, conn_model=gradient_boost_classifier(conn_ft, label, random_state=42)
    evaluate_ml_model_results(conn_predictions, conn_y_test, conn_model, conn_ft, label)
    print('/////////////////////////')

    comb_predictions, comb_y_test, comb_model=gradient_boost_classifier(comb_ft, label, random_state=42)
    evaluate_ml_model_results(comb_predictions, comb_y_test, comb_model, comb_ft, label)



def gradient_boost_classifier(feature_data, label, test_size=0.3, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        feature_data, label, test_size=test_size, random_state=random_state)
    model = GradientBoostingClassifier(random_state=random_state)
    start_time=datetime.now()
    model.fit(x_train, y_train)
    print(datetime.now()-start_time)

    start_time = datetime.now()
    predictions = model.predict(x_test)
    print(datetime.now() - start_time)
    print(model.score(x_test,y_test))
    return predictions, y_test, model

def evaluate_ml_model_results(predictions, y_test, model, feature_data, label):
    start_time = datetime.now()
    auc_cv_score = cross_val_score(model, feature_data, label, cv=10, scoring='roc_auc')
    print(datetime.now() - start_time)
    start_time = datetime.now()
    acc_cv_score = cross_val_score(model, feature_data, label, cv=10, scoring='accuracy')
    print(datetime.now() - start_time)
    print("=== Confusion Matrix ===")
    start_time = datetime.now()
    cf_matrix = confusion_matrix(y_test, predictions)
    print(datetime.now() - start_time)
    print(cf_matrix)
    print('\n')
    print("=== Classification Report ===")
    start_time = datetime.now()
    class_report = classification_report(y_test, predictions)
    print(datetime.now() - start_time)
    print(class_report)
    print('\n')
    print("=== All AUC Scores ===")
    print(auc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", auc_cv_score.mean())
    print('\n')
    print("=== All ACC Scores ===")
    print(acc_cv_score)
    print('\n')
    print("=== Mean ACC Score ===")
    print("Mean ACC Score - Random Forest: ", acc_cv_score.mean())

if __name__ == '__main__':
    main()