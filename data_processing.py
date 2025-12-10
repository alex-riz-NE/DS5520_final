import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys


# Feature lists (EDA)
NUM_FEATURES = ['telecommuting','has_company_logo','has_questions',
    'description_len','requirements_len','company_profile_len', 'description_vocab']
CAT_FEATURES = ['employment_type','function','required_experience','required_education','industry_grouped','country_grouped']
TEXT_FEATURES = ['description','requirements','company_profile']

#####  Missing/Duplicate Value Handling 
def handle_missing_values(df):
    df = df.copy()
    print("Handling missing values...")
    for col in NUM_FEATURES:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(0)
                print(f"  - {col}: filled {n_missing} missing values with 0")
    for col in ['employment_type', 'function', 'required_experience', 'required_education']:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna('Missing')
                f"  - {col}: filled {n_missing} missing values with the object  'Missing'"

    print("Done.\n")
    return df

def drop_exact_duplicates(df,verbose=True):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f'Dropped {before-len(df)} duplicate rows')
    return df

#### Feature Engineering
def add_text_features(df):
    df = df.copy()
    df['description_len'] = df['description'].fillna('').str.len()
    df['requirements_len'] = df['requirements'].fillna('').str.len()
    df['company_profile_len'] = df['company_profile'].fillna('').str.len()
    df['description_vocab'] = df['description'].fillna('').apply(lambda x: len(set(x.lower().split())))
    return df

def group_industry(df,top_n=35):
    df = df.copy()
    top = df['industry'].value_counts().nlargest(top_n).index
    df['industry_grouped'] = (df['industry'].where(df['industry'].isin(top),'Other').fillna('Missing'))
    return df

def group_country(df,top_n=20):
    df = df.copy()
    top = df['country'].value_counts().nlargest(top_n).index
    df['country_grouped'] = ( df['country'].where(df['country'].isin(top),'Other').fillna('Missing'))
    return df

def extract_country(df):
    df = df.copy()
    if 'location' in df.columns:
        df['country'] = (df['location'].fillna('').str.split(',',expand=True)[0].replace('',np.nan) )
    return df


##### Sanity Checks
def sanity_checks(df):
    nulls = df[NUM_FEATURES+CAT_FEATURES].isna().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        raise ValueError(f'Unexpected NaNs detected:\n{bad}')
    return True


def build_feature_matrix(df,verbose=True):
    df = df.copy()
    df = drop_exact_duplicates(df,verbose)
    df = handle_missing_values(df)
    df = extract_country(df)
    df = add_text_features(df)
    df = group_industry(df)
    df = group_country(df)
    sanity_checks(df)
    return df


def build_preprocessor():
    num_pipe = Pipeline([
        ('scaler',StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ])
    return ColumnTransformer([
        ('num',num_pipe,NUM_FEATURES),
        ('cat',cat_pipe,CAT_FEATURES)
    ])

##### Runner Function
def process(path="fake_job_postings.csv", verbose=True):
    jobs = pd.read_csv(path)
    jobs_proc = build_feature_matrix(jobs)
    pre = build_preprocessor()
    X = pre.fit_transform(jobs_proc)

    if verbose:
        print('Preprocessing complete')
        print(f'Loaded data: {jobs.shape}')
        print(f'After preprocessing: {jobs_proc.shape}')
        print(f'Final feature matrix shape: {X.shape}')

    return jobs_proc, X

if __name__ == "__main__":

    jobs_proc, X = process()