"""
Patient Splitting Module ("No Cheating").
Ensures strict patient-level separation across train, validation, and test splits
using GroupKFold / GroupShuffleSplit on study_id to avoid data leakage.
"""
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

class PatientDataSplitter:
    """
    Guarantees that all slices, series, and conditions for any single patient (study_id)
    stay strictly together in either the train, validation, or test pile.
    """
    def __init__(self, n_splits: int = 5, test_size: float = 0.20, seed: int = 42):
        self.n_splits = n_splits
        self.test_size = test_size
        self.seed = seed

    def split_patients(
        self, 
        df: pd.DataFrame, 
        patient_col: str = 'study_id'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataframe into Train (70%), Val (15%), and Test (15%) splits
        with zero patient overlap.
        """
        unique_patients = df[patient_col].unique()
        
        # Step 1: Split into Train+Val vs Test
        splitter_test = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        train_val_idx, test_idx = next(splitter_test.split(df, groups=df[patient_col]))
        
        df_train_val = df.iloc[train_val_idx].copy().reset_index(drop=True)
        df_test = df.iloc[test_idx].copy().reset_index(drop=True)

        # Step 2: Split Train+Val into Train and Val (e.g., 80% train, 20% val of train_val)
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=self.seed)
        train_idx, val_idx = next(splitter_val.split(df_train_val, groups=df_train_val[patient_col]))

        df_train = df_train_val.iloc[train_idx].copy().reset_index(drop=True)
        df_val = df_train_val.iloc[val_idx].copy().reset_index(drop=True)

        # Integrity assertion: No patient overlap
        train_pts = set(df_train[patient_col])
        val_pts = set(df_val[patient_col])
        test_pts = set(df_test[patient_col])

        assert len(train_pts.intersection(val_pts)) == 0, "Data Leakage! Train and Val share patients."
        assert len(train_pts.intersection(test_pts)) == 0, "Data Leakage! Train and Test share patients."
        assert len(val_pts.intersection(test_pts)) == 0, "Data Leakage! Val and Test share patients."

        print(f"[Patient Splitter] Patient integrity verified!")
        print(f"  Train patients: {len(train_pts)} ({len(df_train)} samples)")
        print(f"  Val patients:   {len(val_pts)} ({len(df_val)} samples)")
        print(f"  Test patients:  {len(test_pts)} ({len(df_test)} samples)")

        return df_train, df_val, df_test

    def get_kfold_splits(
        self, 
        df: pd.DataFrame, 
        patient_col: str = 'study_id'
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates K-Fold cross validation splits grouped by patient.
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        splits = []
        for train_idx, val_idx in gkf.split(df, groups=df[patient_col]):
            splits.append((
                df.iloc[train_idx].copy().reset_index(drop=True),
                df.iloc[val_idx].copy().reset_index(drop=True)
            ))
        return splits
