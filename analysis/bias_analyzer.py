from analysis.metrics_analyzer import MetricsAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class BiasAnalyzer(MetricsAnalyzer):

    def __init__(self, log_file_path, metrics_name, t=[], v=[]):
        super(BiasAnalyzer, self).__init__(log_file_path, metrics_name, t, v)

    def plot(self):
        if len(self.metrics_value) > 0:
            df = pd.read_csv('C:/Users/david/Documents/Mscac/CSC-2547/gumbel_5_maxlen_16_vocabsize.csv')
            t = np.array(df['Epoch'])
            v = np.array(df['Bias'])
            plt.plot(self.timestep[:100], self.metrics_value[:100], label='RebarGAN')
            plt.plot(t[:100], v[:100], label='GumbelGAN')
            plt.legend()
            plt.xticks(np.arange(min(self.timestep), max(self.timestep) - 10, 20))
            plt.title(self.metrics_name + ' Graph')
            plt.xlabel('Epoch')
            plt.ylabel(self.metrics_name)
            plt.savefig(self.metrics_name + '_' + self.log_file_path.split('log/')[1] + '.png')
            plt.clf()
            # print average bias
            avgBias = np.mean(self.metrics_value)
            print("Average Bias for " + self.log_file_path.split('log/')[1] + "  = " + str(avgBias))
        else:
            print(self.metrics_name + " not tested for " + " not tested for " + self.log_file_path.split('log/')[1])
