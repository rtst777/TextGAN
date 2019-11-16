from analysis.metrics_analyzer import MetricsAnalyzer
import numpy as np
import matplotlib.pyplot as plt


class LossAnalyzer(MetricsAnalyzer):

    def __init__(self, log_file_path, metrics_name):
        super(LossAnalyzer, self).__init__(log_file_path, metrics_name)

    def plot(self):
        if len(self.metrics_value) > 0:
            plt.plot(self.timestep, self.metrics_value)
            plt.xticks(np.arange(min(self.timestep), max(self.timestep) + 1, 1.0))
            plt.title(self.metrics_name + ' graph')
            plt.xlabel('Epoch')
            plt.ylabel(self.metrics_name)
            plt.show()
        else:
            print(self.metrics_name + " Not Tested")
