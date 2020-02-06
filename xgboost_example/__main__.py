import logging
import warnings

import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Set logger + logging level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Ignore warnings
warnings.filterwarnings("ignore")
# set seed for reproducability
np.random.seed(40)


def load_data():
    """
    Download and prepare the wine classification data set.

    Returns
    -------
    a pandas dataframe containing the wine class data set.
    """
    d = load_wine()
    data = {colname: d.data[:, i] for i, colname in enumerate(d.feature_names)}
    data["target"] = d.target
    return pd.DataFrame(data)


def prep_data(df):
    """
    Prepare wine data set by splitting in X and y and split those into training and test sets.

    Parameters
    ----------
    df: pd.DataFrame
        Data to be split into test and train
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    y = df.target
    X = df.drop(["target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def eval_metrics(y, pred):
    """
    Calculate the percentage of incorrectly predicted label values

    Parameters
    ----------
    y: np.array
        array of label values
    pred: np.arrayt
        array of predicted values

    Returns
    -------
    classification error, a float
    """
    classification_error = np.sum(pred != y) / float(y.shape[0])
    return classification_error


if __name__ == "__main__":

    logger.debug("Create data")
    X_train, X_test, y_train, y_test = prep_data(load_data())
    logger.debug("Converting df to xgb matrix")
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    logger.debug("Initializing model")
    param = {
        "objective": "multi:softmax",
        "eta": 0.1,
        "max_depth": 6,
        "silent": 0,
        "nthread": 3,
        "num_class": 3
    }
    num_round = 500
    # Init an MLflow run
    with mlflow.start_run() as run:
        logger.debug("Training model")
        bst = xgb.train(param, xg_train, num_round)
        logger.debug("Evaluating model")
        pred = bst.predict(xg_test)
        report = classification_report(y_test, pred, output_dict=True)
        classification_error = eval_metrics(y_test, pred)

        # Log all results and parameters to mlflow
        for i in range(len(y_test.unique())):
            num = str(i)
            mlflow.log_metric(f"{num}_precision", report[num]["precision"])
            mlflow.log_metric(f"{num}_recall", report[num]["recall"])
            mlflow.log_metric(f"{num}_f1-score", report[num]["f1-score"])
        mlflow.log_metric("classification_error", classification_error)
        mlflow.log_params(param)
        artifact_location = "artifacts"
        mlflow.xgboost.log_model(bst, artifact_path=artifact_location)
