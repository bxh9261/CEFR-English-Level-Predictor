import os
import pandas as pd


if __name__ == "__main__":
    data = []
    for level in os.listdir("icnale_country_sorted_data")[0:]:
        print(level + "pre")
        if(len(level) == 2 or len(level) == 3):
            print(level)
            level_dir = os.path.join("icnale_country_sorted_data", level)
            for text_file in os.listdir(level_dir):
                try:
                    with open(os.path.join(level_dir, text_file), "r") as f:
                        text = f.read()
                    data.append({"text": text, "label": level})
                except UnicodeDecodeError:
                    print(level_dir, text_file)

    print(len(data))

    df = pd.DataFrame(data)
    df.to_csv(os.path.join("icnale_country_sorted_data", "cefr_leveled_texts.csv"), index=False)
