import pandas as pd
from typing import Optional, Dict

    


def category_weighted_benchmark_score(
    normalise_to_model: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    **score_dfs: pd.DataFrame,
):
    """Weighted scoring for molecular benchmarks from multiple input score DataFrames."""
    if weights is None:
        weights = {name: 1.0 for name in score_dfs.keys()}

    scores = {}
    # Only consider DataFrames that are not None
    valid_dfs = {name: df for name, df in score_dfs.items() if df is not None}
    if not valid_dfs:
        return pd.DataFrame()
    # Find intersection of all model/method names across all input DataFrames
    all_models = set.intersection(
        *(set(df["Model"] if "Model" in df.columns else df["Method"]) for df in valid_dfs.values())
    )

    for model in all_models:
        entry = {}
        total = 0.0
        denom = 0.0
        for name, df in score_dfs.items():
            if df is None:
                continue
            col = "Model" if "Model" in df.columns else "Method"
            # Try to find a score column with "Score" in name, prefer "\u2193" if present
            score_col = None
            for c in df.columns:
                if "Score" in c:
                    score_col = c
                    if "\u2193" in c:
                        break
            if score_col is None:
                continue
            # Only add if model present in this df
            if model not in set(df[col]):
                continue
            score = df.loc[df[col] == model, score_col].values[0]
            entry[f"{name.capitalize()} {score_col}"] = score
            total += weights.get(name, 1.0) * score
            #denom += weights.get(name, 1.0)
        entry["Avg MAE \u2193"] = total #/ denom if denom > 0 else None
        scores[model] = entry

    df = pd.DataFrame.from_dict(scores, orient="index").reset_index().rename(columns={"index": "Model"})

    print(df)

    if normalise_to_model:
        norm_val = df.loc[df["Model"] == normalise_to_model, "Avg MAE \u2193"].values[0]
        df["Avg MAE \u2193"] = df["Avg MAE \u2193"] / norm_val


    df = df.round(3).sort_values(by="Avg MAE \u2193").reset_index(drop=True)
    df["Rank"] = df["Avg MAE \u2193"].rank(ascending=True)
    
    
    return df