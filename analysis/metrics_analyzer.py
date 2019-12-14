import re
import numpy as np

class MetricsAnalyzer:
    def __init__(self, log_file_path, metrics_name, t, v):
        self.metrics_name = metrics_name
        self.log_file_path = log_file_path
        self.timestep, self.metrics_value = self._parse_log_file(log_file_path, metrics_name, t, v)

    def _parse_log_file(self, log_file, metrics_name, t, v):

        timeList = []
        metricsList = []
        metricsArr = np.empty((0, 4))
        timePatternMLE = r'epoch_mle (\d+)'
        if t == [] and metrics_name == 'BLEU-[2, 3, 4, 5]':
            timePattern = r'epoch (\d+)'
        else:
            timePattern = r'ADV EPOCH (\d+)'
        #timePattern = r'ADV EPOCH (\d+)'
        metricsPattern = r'.' + re.escape(metrics_name) + ' = ' + '([+-]?([0-9]*[.])?[0-9]+)'
        if metrics_name == 'BLEU-[2, 3, 4, 5]':
            metricsPattern = r'.' + re.escape(metrics_name) + ' = ' + '\[(.*?)\]'
        if metrics_name == 'Bias':
            metricsPattern = r'tensor\(([+-]?([0-9]*[.])?[0-9]+)'

        with open(log_file, 'r') as file:
            for line in file:
                timeMatch = re.search(timePattern, line)
                timeMLEMatch = re.search(timePatternMLE, line)
                metricsMatch = re.search(metricsPattern, line)
                if timeMLEMatch:
                    timeList.append(int(timeMLEMatch.group(1)))
                if timeMatch:
                    if metrics_name == 'BLEU-[2, 3, 4, 5]':
                        timeList.append(int(timeMatch.group(1)) + 5) # 5 pretrain epochs
                    else:
                        timeList.append(timeMatch.group(1))
                if metricsMatch:
                    if metrics_name == 'BLEU-[2, 3, 4, 5]':
                        metricsList = np.reshape(np.array(list(map(float, metricsMatch.group(1).split(", ")))), (1, 4))
                        metricsArr = np.append(metricsArr, metricsList, axis=0)
                    else:
                        metricsList.append(metricsMatch.group(1))

        timeArr = np.array(timeList)
        timeArr = timeArr.astype(np.int)
        if metrics_name != 'BLEU-[2, 3, 4, 5]':
            metricsArr = np.array(metricsList)
            metricsArr = metricsArr.astype(np.float)

        #print(np.shape(timeArr))
        #print(np.shape(metricsArr))


        return timeArr, metricsArr

    def plot(self):
        pass