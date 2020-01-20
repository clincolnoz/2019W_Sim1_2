# -*- coding: utf-8 -*-
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix


# Loading data
def load_data():
    audio_test = pd.read_csv(data_path / "audio_predictions_labelled_test.csv")
    audio_train = pd.read_csv(data_path / "audio_predictions_labelled_train.csv")
    image_test = pd.read_csv(data_path / "image_predictions_labelled_test.csv")
    image_train = pd.read_csv(data_path / "image_predictions_labelled_train.csv")

    audio = pd.concat([audio_test, audio_train], ignore_index=True)
    image = pd.concat([image_test, image_train], ignore_index=True)

    full_df = pd.concat([audio, image], axis=1, ignore_index=True)

    full_df.columns = ["probs_kermit_audio", "probs_no_kermit_audio", "true_audio", "pred_audio",
                       "probs_kermit_image", "probs_no_kermit_image", "true_image", "pred_image"]

    full_df["label"] = full_df.true_audio | full_df.true_image

    full_df = full_df.drop(columns=["true_audio", "pred_audio", "true_image", "pred_image"])

    full_df = full_df.dropna()

    full_x = full_df.drop(columns="label").values.astype(float)
    full_y = full_df["label"].values.astype(int)

    # Train/Test split
    train_x, test_x, train_y, test_y = train_test_split(
        full_x, full_y,
        test_size=0.2,
        random_state=0,
    )

    return train_x, test_x, train_y, test_y


def build_dnn_cb(activation="relu"):
    model = Sequential()

    model.add(Dense(15, activation=activation, input_dim=4))  # TODO(REMOVE HARDCODED train.shape[1])
    model.add(Dropout(0.2))
    model.add(Dense(7, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation=activation))

    model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def plot_roc(y_test, y_pred):
    """

    :param y_test:
    :param y_pred:
    :param auc_rf:
    """
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


def main():
    logger = logging.getLogger(__name__)
    logger.info('Loaded data and preparing timeline')

    train_x, test_x, train_y, test_y = load_data()

    pipe = Pipeline(
        [
            [
                "scaler",
                StandardScaler()
            ],
            [
                "classifier",
                # RandomForestClassifier(),
                KerasClassifier(build_dnn_cb, epochs=10),
            ],
        ]
    )

    # Grid Search
    param_grid = {
        # 'classifier__n_estimators': (50, 100, 400, 1000),
        # "classifier__criterion": ("gini", "entropy"),
        # 'classifier__max_depth': (2, 5, 8, None),
        # 'classifier__bootstrap': (True, False),
    }

    gs = GridSearchCV(
        pipe,
        param_grid,
        scoring="accuracy",
        cv=5,
        verbose=1,
        n_jobs=1,
    )

    gs.fit(train_x, train_y)
    pipe = gs.best_estimator_
    pred_y = pipe.predict(test_x)

    print(f"Accuracy: {accuracy_score(test_y, pred_y)}")
    print(f"Recall:   {recall_score(test_y, pred_y)}")
    print(f"F1:       {f1_score(test_y, pred_y)}")
    print(f"Precision {precision_score(test_y, pred_y)}")

    # plot_roc(test_y, pred_y)
    print(confusion_matrix(test_y, pred_y))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    root_path = Path(__file__).resolve().parents[1]
    data_path = root_path / "data"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
