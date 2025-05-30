import pandas as pd


def preprocess_breast_cancer(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessor for Dataset #1"""
    target = "diagnosis"

    df[target] = df[target].map({"M": 0, "B": 1})
    df = df.drop(columns=["id"])

    return df


def preprocess_car_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessor for Dataset #2"""
    df["class"] = df["class"].map(
        {
            "unacc": 0,
            "acc": 1,
            "good": 2,
            "vgood": 3,
        }
    )

    df["doors"] = df["doors"].map({"2": 2, "3": 3, "4": 4, "5more": 5})

    high_map = {"low": 0, "med": 1, "high": 2, "vhigh": 3}

    df["buying"] = df["buying"].map(high_map)
    df["safety"] = df["safety"].map(high_map)
    df["maint"] = df["maint"].map(high_map)

    df["persons"] = df["persons"].map({"2": 2, "4": 4, "more": 6})

    df["lug_boot"] = df["lug_boot"].map({"small": 0, "med": 1, "big": 2})

    return df


def preprocess_student_performance_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessor for Dataset #3"""
    lmh = {
        "Low": -1,
        "Medium": 0,
        "High": +1,
    }

    yn = {
        "Yes": +1,
        "No": -1,
    }

    df = df.dropna(subset=["Teacher_Quality"])

    df["Parental_Involvement"] = df["Parental_Involvement"].map(lmh)
    df["Access_to_Resources"] = df["Access_to_Resources"].map(lmh)
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map(yn)
    df["Motivation_Level"] = df["Motivation_Level"].map(lmh)
    df["Internet_Access"] = df["Internet_Access"].map(yn)
    df["Family_Income"] = df["Family_Income"].map(lmh)
    df["Teacher_Quality"] = df["Teacher_Quality"].map(lmh)
    df["School_Type"] = df["School_Type"].map(
        {
            "Public": +1,
            "Private": -1,
        }
    )
    df["Peer_Influence"] = df["Peer_Influence"].map(
        {
            "Positive": +1,
            "Neutral": 0,
            "Negative": -1,
        }
    )
    df["Learning_Disabilities"] = df["Learning_Disabilities"].map(yn)
    df["Parental_Education_Level"] = (
        df["Parental_Education_Level"]
        .map(
            {
                "Postgraduate": +3,
                "College": +2,
                "High School": +1,
            }
        )
        .fillna(0)
    )
    df["Distance_from_Home"] = (
        df["Distance_from_Home"]
        .map(
            {
                "Near": +1,
                "Moderate": 0,
                "Far": -1,
            }
        )
        .fillna(0)
    )
    df["Gender"] = (
        df["Gender"]
        .map(
            {
                "Female": +1,
                "Male": -1,
            }
        )
        .fillna(0)
    )

    return df
