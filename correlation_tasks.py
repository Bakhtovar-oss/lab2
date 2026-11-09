"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.path, sep=r'\s+')
    
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    correlations = {}
    strongest_mri_feature = None
    max_abs_corr = -1
    
    for gender, group in df.groupby('Gender'):
        correlations[gender] = {}
        for feature in features:
            corr = group[feature].corr(group['MRI_Count'])
            correlations[gender][feature] = corr
            
            if pd.notna(corr) and abs(corr) > max_abs_corr:
                max_abs_corr = abs(corr)
                strongest_mri_feature = feature
                
    return BrainCorrelationSummary(
        correlations=correlations,
        strongest_mri_feature=strongest_mri_feature
    )
