from textstat import textstat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, LabelEncoder, StandardScaler, RobustScaler, robust_scale
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

LABELS = ["A2", "B1", "B2", "C2"]

METRICS = [
    textstat.flesch_reading_ease,
    textstat.smog_index,
    textstat.flesch_kincaid_grade,
    textstat.coleman_liau_index,
    textstat.automated_readability_index,
    textstat.dale_chall_readability_score,
    textstat.difficult_words,
    textstat.linsear_write_formula,
    textstat.gunning_fog,
    textstat.text_standard,
]


class Predictor:
    def __init__(self, prediction_function):
        self.predict_func = prediction_function
        self.scaler = RobustScaler(with_centering=False)

    def predict(self, X):
        output = X.apply(self._predict_text)
        print(output)
        output = pd.DataFrame(output)
        output = (robust_scale(output))
        scaled_outputs = pd.DataFrame(self.scaler.fit_transform(output)+1.5)
        print(scaled_outputs)
        roundscale = [round(p) for p in scaled_outputs[0]]
        #for idx, highest in enumerate(printer):
           # if highest == 3:
                #print(X[idx])
        print(roundscale)
        return roundscale

    def get_name(self):
        return self.predict_func.__name__

    def _predict_text(self, text):
        if self.get_name() == "text_standard":
            return self.predict_func(text, float_output=True)
        else:
            return self.predict_func(text)


def load_data():
    test = pd.read_csv("../icnale4_data/test.csv")
    X = test.text
    y = test.label
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return X, y


def calculate_metrics(X, y):
    for metric in METRICS:
        predictor = Predictor(metric)
        preds = predictor.predict(X)
        score2 = confusion_matrix(y, preds, labels=[0,1,2,3])
        score1 = accuracy_score(y, preds)
        score = f1_score(y, preds, labels=[0,1,2,3], average="weighted")
        print(f"{predictor.get_name()} F1: {score}")
        print(f"{predictor.get_name()}:\n {score2}")
        print(f"{predictor.get_name()} Accuracy:\n {score1}")


if __name__ == "__main__":
    X, y = load_data()
    print(f"baseline random: {1 / len(LABELS)}")
    calculate_metrics(X, y)