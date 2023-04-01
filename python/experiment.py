import pandas as pd


class Experiment:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.subjects = {}

    @classmethod
    def from_csv(cls, filepath):
        experiment = cls(pd.read_csv(filepath))
        experiment.update_subjects()
        return experiment

    def copy(self):
        # do a deep copy
        copy = Experiment(self.df.copy())
        copy.subjects = self.subjects.copy()
        return copy

    def update_subjects(self, column_markers=("Group Name", "Subject Name")):
        for group in self.df[column_markers[0]].drop_duplicates():
            self.subjects[group] = [s for s in
                                    self.df[column_markers[1]][self.df[column_markers[0]] == group].drop_duplicates()]

    def to_csv(self, filepath):
        self.df.to_csv(filepath, index=False)
