# truthmindr_agent/app/explain.py
import pandas as pd
from tools.rephraser import rephrase_result


def explain_from_csv(path="outputs/enriched.csv", limit=5):
    df = pd.read_csv(path)
    for i, row in df.head(limit).iterrows():
        explanation = rephrase_result(row.to_dict())
        print(f"🔹 Post ID: {row['post_id']}")
        print(f"   Title: {row['clean_title']}")
        print(f"   👉 Explanation: {explanation}\n")


if __name__ == "__main__":
    explain_from_csv()
