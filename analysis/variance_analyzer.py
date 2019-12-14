from analysis.metrics_analyzer import MetricsAnalyzer
import numpy as np
import matplotlib.pyplot as plt


class VarianceAnalyzer(MetricsAnalyzer):

    def __init__(self, log_file_path, metrics_name, t, v):
        super(VarianceAnalyzer, self).__init__(log_file_path, metrics_name, t, v)
        self.t = t
        self.v = v

    def plot(self):
        if len(self.metrics_value) > 0:
            plt.plot(self.timestep, self.metrics_value, label='GumbelGAN')
            plt.plot(self.t, self.v, label='RebarGAN')
            plt.legend()
            plt.xticks(np.arange(min(self.timestep), max(self.timestep) + 2, 20))
            #plt.title(self.metrics_name + ' Graph')
            plt.title('Variance Graph')
            plt.xlabel('Epoch')
            #plt.ylabel(self.metrics_name)
            plt.ylabel('Log Variance')
            plt.savefig(self.metrics_name + '_' + self.log_file_path.split('log/')[1] + '_var.pdf')
            plt.clf()
            # print average variance
            avgVariance = np.mean(self.metrics_value)
            print("Average Variance for " + self.log_file_path.split('log/')[1] + "  = " + str(avgVariance))
        else:
            print(self.metrics_name + " not tested for " + " not tested for " + self.log_file_path.split('log/')[1])
