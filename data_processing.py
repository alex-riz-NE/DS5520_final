import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ============================================================
# Feature lists (EDA)
# ============================================================

NUM_FEATURES = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "description_len",
    "requirements_len",
    "company_profile_len",
    "description_vocab",
]

CAT_FEATURES = [
    "employment_type",
    "function",
    "required_experience",
    "required_education",
    "industry_grouped",
    "country_grouped",
]

TEXT_FEATURES = [
    "description",
    "requirements",
    "company_profile",
]


# ============================================================
# Missing / duplicate handling
# ============================================================

def drop_exact_duplicates(df, verbose=True):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"Dropped {before - len(df)} duplicate rows")
    return df


def handle_missing_values(df):
    df = df.copy()

    for col in NUM_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in [
        "employment_type",
        "function",
        "required_experience",
        "required_education",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    for col in TEXT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


# ============================================================
# Feature engineering
# ============================================================

def extract_country(df):
    df = df.copy()
    if "location" in df.columns:
        df["country"] = (
            df["location"]
            .fillna("")
            .str.split(",", expand=True)[0]
            .replace("", np.nan)
        )
    return df


def group_industry(df, top_n=35):
    df = df.copy()
    top = df["industry"].value_counts().nlargest(top_n).index
    df["industry_grouped"] = (
        df["industry"]
        .where(df["industry"].isin(top), "Other")
        .fillna("Missing")
    )
    return df


def group_country(df, top_n=20):
    df = df.copy()
    top = df["country"].value_counts().nlargest(top_n).index
    df["country_grouped"] = (
        df["country"]
        .where(df["country"].isin(top), "Other")
        .fillna("Missing")
    )
    return df


def add_text_features(df):
    df = df.copy()

    df["description_len"] = df["description"].str.len()
    df["requirements_len"] = df["requirements"].str.len()
    df["company_profile_len"] = df["company_profile"].str.len()

    df["description_vocab"] = (
        df["description"]
        .str.lower()
        .str.split()
        .apply(lambda x: len(set(x)) if isinstance(x, list) else 0)
    )

    return df


# ============================================================
# Sanity checks
# ============================================================

def sanity_checks(df):
    nulls = df[NUM_FEATURES + CAT_FEATURES].isna().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        raise ValueError(f"Unexpected NaNs detected:\n{bad}")
    return True


# ============================================================
# High-level preprocessing
# ============================================================

def build_feature_matrix(df, verbose=True):
    df = df.copy()

    df = drop_exact_duplicates(df, verbose)
    df = extract_country(df)
    df = handle_missing_values(df)
    df = add_text_features(df)
    df = group_industry(df)
    df = group_country(df)
    sanity_checks(df)

    return df


# ============================================================
# Structured data pipeline
# ============================================================

def build_structured_preprocessor():
    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])


def get_structured_matrix(df, preprocessor=None):
    if preprocessor is None:
        preprocessor = build_structured_preprocessor()
        X = preprocessor.fit_transform(df)
    else:
        X = preprocessor.transform(df)

    return X, preprocessor


# ============================================================
# Text pipeline (TF-IDF via sklearn)
# ============================================================

def build_tfidf_vectorizer(
    max_features=5000,
    min_df=5,
    stop_words="english",
    ngram_range=(1, 2),
):
    return TfidfVectorizer(
        #max_features=max_features,
        min_df=min_df,
        stop_words=stop_words,
        ngram_range=ngram_range,
    )


def get_tfidf_matrix(df, vectorizer=None):
    corpus = (
        df[TEXT_FEATURES]
        .astype(str)
        .agg(" ".join, axis=1)
    )

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()
        X = vectorizer.fit_transform(corpus)
    else:
        X = vectorizer.transform(corpus)

    return X, vectorizer


# ============================================================
# Train / test split
# ============================================================

def split_train_test(X, y, test_size=0.3, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ============================================================
# Runner function (matches your style)
# ============================================================

def process(
    path="fake_job_postings.csv",
    verbose=True,
    return_text=True,
):
    jobs = pd.read_csv(path)
    jobs_proc = build_feature_matrix(jobs, verbose)

    X_struct, struct_pre = get_structured_matrix(jobs_proc)
    X_text, tfidf_vec = get_tfidf_matrix(jobs_proc)

    y = jobs_proc["fraudulent"].values if "fraudulent" in jobs_proc.columns else None

    if verbose:
        print("Preprocessing complete")
        print(f"Loaded data: {jobs.shape}")
        print(f"After preprocessing: {jobs_proc.shape}")
        print(f"Structured matrix: {X_struct.shape}")
        print(f"TF-IDF matrix: {X_text.shape}")

    return {
        "jobs_proc": jobs_proc,
        "X_struct": X_struct,
        "X_text": X_text if return_text else None,
        "y": y,
        "structured_preprocessor": struct_pre,
        "tfidf_vectorizer": tfidf_vec,
    }


# ============================================================
# CLI test hook
# ============================================================

if __name__ == "__main__":
    process()
