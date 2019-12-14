from analysis.metrics_analyzer import MetricsAnalyzer
import numpy as np
import matplotlib.pyplot as plt


class BleuAnalyzer(MetricsAnalyzer):

    def __init__(self, log_file_path, metrics_name, t, v):
        super(BleuAnalyzer, self).__init__(log_file_path, metrics_name, t, v)
        self.t = t
        self.v = v

    def plot(self):
        if len(self.metrics_value) > 0:
            for i in range(4):
                plt.plot(self.timestep, self.metrics_value[:, i], label='ReLbarGAN')
                if self.t != []:
                    plt.plot(self.t, self.v[:, i], label='ReLGAN')
                plt.legend()
                meanVal = (self.v[1, i] + self.metrics_value[1, i])/2
                print(meanVal)
                plt.xticks(np.arange(min(self.timestep), max(self.timestep) + 200, 200))
                if i == 0:
                    plt.axvline(x=5, ymin=meanVal - 0.52, ymax=meanVal - 0.32, linestyle='--', color='black')
                elif i == 1:
                    plt.axvline(x=5, ymin=meanVal - 0.45, ymax=meanVal - 0.25, linestyle='--', color='black')
                elif i == 2:
                    plt.axvline(x=5, ymin=meanVal - 0.22, ymax=meanVal - 0.02, linestyle='--', color='black')
                else:
                    plt.axvline(x=5, ymin=meanVal - 0.15, ymax=meanVal + 0.05, linestyle='--', color='black')
                #plt.title(self.metrics_name + ' graph')
                plt.title('Bleu ' + str(i + 2) + ' Score for ReLGAN and ReLbarGAN')
                plt.xlabel('Epoch')
                plt.ylabel('Bleu ' + str(i + 2))
                plt.savefig('Bleu ' + str(i + 2) + '_' + self.log_file_path.split('log/')[1] + '_new.pdf')
                plt.clf()
            # print average bias
            # avgBias = np.mean(self.metrics_value)
            # print("Average Bias for " + self.log_file_path.split('log/')[1] + "  = " + str(avgBias))
        else:
            print(self.metrics_name + " not tested for " + " not tested for " + self.log_file_path.split('log/')[1])
