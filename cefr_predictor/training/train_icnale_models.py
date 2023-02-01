import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from preprocessing import generate_features

RANDOM_SEED = 0

features = ["complexity", "pos", "syntax"]
directories = ["icnale4_data", "crow_data"]

label_encoder = None

def train(model, feature, directory):
    X_train, y_train = load_data(f"../../{directory}/train.csv")
    X_test, y_test = load_data(f"../../{directory}/test.csv")

    print(f"Training {model['name']}, for {directory} with {feature} features!")
    pipeline = build_pipeline(model["model"], feature)
    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))
    save_model(pipeline, model["name"], feature, directory)


def build_pipeline(model, feature):
    """Creates a pipeline with feature extraction, feature scaling, and a predictor."""
    return Pipeline(
        steps=[
            ("generate features", FunctionTransformer(generate_features, kw_args = {"type": feature})),
            ("scale features", StandardScaler()),
            ("model", model),
        ],
        verbose=True,
    )


def load_data(path_to_data):
    data = pd.read_csv(path_to_data)
    X = data.text.tolist()
    y = encode_labels(data.label)
    return X, y


def encode_labels(labels):
    global label_encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return label_encoder.transform(labels)


def save_model(model, name, feature, directory):
    name = name.lower().replace(" ", "_")
    file_name = f"../../cefr_predictor/models/featuremodels/{name}_{directory}_{feature}.joblib"
    print(f"Saving {file_name}")
    dump(model, file_name)


models = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(random_state=RANDOM_SEED),
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(random_state=RANDOM_SEED),
    },
    {"name": "SVC", "model": SVC(random_state=RANDOM_SEED, probability=True)},
]

if __name__ == "__main__":
    for directory in directories:
        for feature in features:
            for model in models:
                train(model, feature, directory)