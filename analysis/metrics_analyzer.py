import re
import numpy as np

class MetricsAnalyzer:
    def __init__(self, log_file_path, metrics_name):
        self.metrics_name = metrics_name
        self.timestep, self.metrics_value = self._parse_log_file(log_file_path, metrics_name)

    def _parse_log_file(self, log_file, metrics_name):

        timeList = []
        metricsList = []
        timePattern = r'ADV EPOCH (\d)'
        metricsPattern = r'.' + re.escape(metrics_name) + ' = ' + '([+-]?([0-9]*[.])?[0-9]+)'

        with open(log_file, 'r') as file:
            for line in file:
                timeMatch = re.search(timePattern, line)
                metricsMatch = re.search(metricsPattern, line)
                if timeMatch:
                    timeList.append(timeMatch.group(1))
                if metricsMatch:
                    metricsList.append(metricsMatch.group(1))

        timeArr = np.array(timeList)
        timeArr = timeArr.astype(np.int)
        metricsArr = np.array(metricsList)
        metricsArr = metricsArr.astype(np.float)

        print(timeArr)
        print(metricsArr)

        return timeArr, metricsArr

    def plot(self):
        pass