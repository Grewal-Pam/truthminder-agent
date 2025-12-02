def enrich_metadata(df):
    if "clean_title" not in df.columns:
        df["clean_title"] = df["title"].astype(str)
    df["title_len"] = df["clean_title"].str.len()
    df["has_image"] = df["image_url"].astype(str).str.startswith(("http://","https://"))
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df
