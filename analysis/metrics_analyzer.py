class MetricsAnalyzer:
    def __init__(self, log_file_path, metrics_name):
        self.metrics_name = metrics_name
        self.timestep, self.metrics_value = self._parse_log_file(log_file_path, metrics_name)

    def _parse_log_file(self, log_file, metrics_name):
        return [], []

    def plot(self):
        pass