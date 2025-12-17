def profile_dataset(df):
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    profile = {
        "num_features": len(num_cols),
        "cat_features": len(cat_cols)
    }
    return profile, num_cols, cat_cols
