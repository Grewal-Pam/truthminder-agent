import os
import pandas as pd
from datetime import datetime

def save_to_parquet(df, layer="bronze", source="unknown"):
    ts = datetime.utcnow().strftime("%Y%m%d")
    path = f"data/lake/{layer}/{source}/date={ts}"

    os.makedirs(path, exist_ok=True)
    file = f"{path}/data.parquet"

    df.to_parquet(file, index=False)
    print(f"📦 [{layer}] wrote {file}")
