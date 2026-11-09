"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.path)
    
    missing_values = df.isna().sum().to_dict()
    older_than_30 = int((df['Age'] > 30).sum())
    
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age_by_pclass = {int(k): v for k, v in mean_age_by_pclass.items()}
    
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival_rate_by_pclass = {int(k): v for k, v in survival_rate_by_pclass.items()}
    
    top_5_fares = df['Fare'].nlargest(5).tolist()
    
    return TitanicSummary(
        missing_values=missing_values,
        older_than_30=older_than_30,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        top_5_fares=top_5_fares
    )
