from experiment import Experiment


class Table:
    @staticmethod
    def generate_pitch_distribution_table(experiment: Experiment):
        return experiment.df.melt(id_vars=["Group", "Subject"],
                                  value_vars=["InitialPitch", "EndingPitch"],
                                  var_name="SamplingTime",
                                  value_name="Pitch")

    @staticmethod
    def generate_pitch_variance_table(experiment: Experiment):
        return Table.generate_pitch_distribution_table(experiment) \
            .groupby(["Group", "Subject", "SamplingTime"], as_index=False)["Pitch"].var() \
            .rename(columns={"Pitch": "PitchVariance"})
