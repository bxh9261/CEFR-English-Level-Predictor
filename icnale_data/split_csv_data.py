import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 0


if __name__ == "__main__":
    data = pd.read_csv("icnale_data/cefr_leveled_texts_with_rst.csv", encoding="utf-8")
    train, test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED, stratify=data.label
    )
    train.to_csv("icnale_data/train_rst.csv", index=False)
    test.to_csv("icnale_data/test_rst.csv", index=False)