from analysis.loss_analyzer import LossAnalyzer
from analysis.variance_analyzer import VarianceAnalyzer
from analysis.bias_analyzer import BiasAnalyzer
from analysis.bleu_analyzer import BleuAnalyzer
import os

def main():
    log_file_paths = [
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_gumbelgan_3_maxlen_7_vocabsize.txt',
        #os.path.dirname(os.path.dirname(__file__)) + '/log/log_gumbelgan_3_maxlen_18_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_gumbelgan_15_maxlen_7_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_gumbelgan_15_maxlen_18_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_gumbelgan_51_maxlen_7_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_gumbelgan_51_maxlen_18_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_3_maxlen_7_vocabsize.txt',
        #os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_3_maxlen_18_vocabsize.txt'#,
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_15_maxlen_7_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_15_maxlen_18_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_51_maxlen_7_vocabsize.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_51_maxlen_18_vocabsize.txt'
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_reLbargan_5_pre.txt',
        # os.path.dirname(os.path.dirname(__file__)) + '/log/log_relgan_5_pre.txt'
        #os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_5_maxlen_16_vocabsize.txt'
        #os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_3_maxlen_18_vocabsize_bias.txt'
        #os.path.dirname(os.path.dirname(__file__)) + '/log/log_rebargan_5_maxlen_16_vocabsize.txt'
        os.path.dirname(os.path.dirname(__file__)) + '/log/log_reLbargan_5_pre_2000_adv.txt',
        os.path.dirname(os.path.dirname(__file__)) + '/log/log_relgan_1211.txt'
    ]

    metrics_analyzers = [
        [LossAnalyzer, 'rebar_loss'],
        [LossAnalyzer, 'g_loss'],
        [VarianceAnalyzer, 'theta_gradient_log_var'],
        [BiasAnalyzer, 'Bias'],
        [BleuAnalyzer, 'BLEU-[2, 3, 4, 5]']
    ]

    #A = BiasAnalyzer(log_file_paths[0], 'Bias')
    #A.plot()

    #A = VarianceAnalyzer(log_file_paths[1], 'theta_gradient_log_var', [], [])
    #t = A.timestep
    #v = A.metrics_value
    #VarianceAnalyzer(log_file_paths[0], 'theta_gradient_log_var', t, v).plot()

    A = BleuAnalyzer(log_file_paths[1], 'BLEU-[2, 3, 4, 5]', [], [])
    t = A.timestep
    v = A.metrics_value
    BleuAnalyzer(log_file_paths[0], 'BLEU-[2, 3, 4, 5]', t, v).plot()

    # for log_file_path in log_file_paths:
    #     for metrics_analyzer in metrics_analyzers:
    #         metrics_analyzer[0](log_file_path, metrics_analyzer[1]).plot()

if __name__ == "__main__":
    main()