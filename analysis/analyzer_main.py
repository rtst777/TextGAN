from analysis.loss_analyzer import LossAnalyzer

def main():
    log_file_paths = [
        "log/example_log.txt",
    ]

    metrics_analyzers = [
        LossAnalyzer,
    ]

    for log_file_path in log_file_paths:
        for metrics_analyzer in metrics_analyzers:
            metrics_analyzer(log_file_path).plot()

if __name__ == "__main__":
    main()