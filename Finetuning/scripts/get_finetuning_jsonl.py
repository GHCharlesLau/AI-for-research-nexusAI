import os
import pandas as pd

data_path = r"../data/"

INPUT_FILES = {
    "train.csv": "train.jsonl",
    "valid.csv": "valid.jsonl",
    "test.csv": "test.jsonl",
}

HEADLINE_COL = "headline"
CTR_COL = "CTR"

SYSTEM_PROMPT = "You are a model that predicts CTR from a headline."

def make_record(row, has_label: bool):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Headline: {row[HEADLINE_COL]}"},
    ]
    if has_label:
        messages.append({"role": "assistant", "content": f"{row[CTR_COL]:.6f}"})
    return {"messages": messages}

def process_file(in_path: str, out_path: str):
    df = pd.read_csv(in_path)

    # preprocess
    df = df.dropna(subset=[HEADLINE_COL])
    df = df.drop_duplicates(subset=[HEADLINE_COL])
    df[HEADLINE_COL] = df[HEADLINE_COL].astype(str).str.strip()
    df = df[df[HEADLINE_COL].str.len() > 0]

    has_label = CTR_COL in df.columns

    # make sure the label is numerical
    if has_label:
        df[CTR_COL] = pd.to_numeric(df[CTR_COL], errors="coerce")
        df = df.dropna(subset=[CTR_COL])

    records = [make_record(row, has_label) for _, row in df.iterrows()]
    pd.DataFrame(records).to_json(out_path, orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    for in_file, out_file in INPUT_FILES.items():
        in_file = os.path.join(data_path, in_file)
        out_file = os.path.join(data_path, out_file)
        if os.path.exists(in_file):
            process_file(in_file, out_file)
            print(f"Converted {in_file} -> {out_file}")
        else:
            print(f"Skip: {in_file} not found")
