import pandas as pd

categorical_columns = [
    "proto",
    "state",
    "service"
]

target = "attack_cat"

drop_columns = [
    "id",
    "ct_ftp_cmd",
    "dloss",
    "sloss",
    "dwin",
    "ct_srv_dst"
]


def load_clean_data(path="networkTraffic.csv"):
    df_clean = pd.read_csv(
        path,
        na_values=["?", " ?"],
        skipinitialspace=True
    )

    df_clean["service"] = df_clean["service"].fillna("missing")

    df_clean = df_clean.drop(columns=drop_columns)

    df_clean.loc[df_clean["is_ftp_login"].isin([2, 4]), "is_ftp_login"] = 1

    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    return df_clean


def get_X_y(df_clean):
    numeric_columns = [
        col for col in df_clean.columns
        if col not in categorical_columns + [target]
    ]

    X = df_clean[categorical_columns + numeric_columns]
    y = df_clean[target]

    return X, y, categorical_columns, numeric_columns
