import os
import json
import pandas as pd


if __name__ == "__main__":
    data = []
    json_extension = ".json"
    for level in os.listdir("icnale_data")[0:]:
        print(level + "pre")
        if(len(level) == 2 or len(level) == 4):
            print(level)
            level_dir = os.path.join("icnale_data", level)
            for text_file in os.listdir(level_dir):
                if(text_file.endswith(".txt")):
                    try:
                        with open(os.path.join(level_dir, text_file), "r") as f:
                            text = f.read()
                        jsonFile = os.path.join(level_dir, text_file) + ".json"
                        try:
                            jsonFileOpen = open(jsonFile)
                            fulljson = json.load(jsonFileOpen)
                            rsttree = fulljson['scored_rst_trees'][0]
                        except OSError:
                            print("Could not open/read file:", jsonFile)
                            rsttree = ""
                        data.append({"text": text, "label": level, "rst-tree": rsttree})
                    except UnicodeDecodeError:
                        print(level_dir, text_file)

    print(len(data))

    df = pd.DataFrame(data)
    df.to_csv(os.path.join("icnale_data", "cefr_leveled_texts_with_rst.csv"), index=False)