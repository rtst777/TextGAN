from analysis.metrics_analyzer import MetricsAnalyzer
import numpy as np
import matplotlib.pyplot as plt


class LossAnalyzer(MetricsAnalyzer):

    def __init__(self, log_file_path, metrics_name, t=[], v=[]):
        super(LossAnalyzer, self).__init__(log_file_path, metrics_name, t, v)

    def plot(self):
        if len(self.metrics_value) > 0:
            plt.plot(self.timestep, self.metrics_value)
            plt.xticks(np.arange(min(self.timestep), max(self.timestep) + 2, 20))
            plt.title(self.metrics_name + ' graph')
            plt.xlabel('Epoch')
            plt.ylabel(self.metrics_name)
            plt.savefig(self.metrics_name + '_' + self.log_file_path.split('log/')[1] + '.png')
            plt.clf()
        else:
            print(self.metrics_name + " not tested for " + self.log_file_path.split('log/')[1])
