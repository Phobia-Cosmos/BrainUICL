ISRUC_DICT = {
    "pretrain_lr": 1e-4,
    "ssl_lr": 1e-6,
    "incremental_lr": 1e-7,
    "batch": 16
}


class ModelConfig(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.ConvDrop = 0.1
        self.EncoderParam = EncoderConfig()
        self.SleepMlpParam = SleepMlpParam()
        self.FaceMlpParam = FaceMlpParam()
        self.BCI2000MlpParam = BCI2000MlpParam()
        self.NumClasses = 5
        self.ClassNames = ['W', 'N1', 'N2', 'N3', 'REM']
        self.ClassNamesFace = ['Anger',
                               'Disgust',
                               'Fear',
                               'Sadness',
                               'Neutral',
                               'Amusement',
                               'Inspiration',
                               'Joy',
                               'Tenderness']
        self.ClassNamesBCI2000 = ['Left', 'Right', 'Fist', 'Feet']
        self.SeqLength = 20
        # TODO:为什么BatchSize是32？
        self.BatchSize = 32
        self.EpochLength = 3000
        self.EpochLengthFace = 7500
        self.EpochLengthBCI2000 = 640
        # TODO:那如果对于其他数据集怎么处理呢？
        channel_num = self.get_channel_info()
        if self.dataset == "ISRUC":
            self.EegNum = channel_num[0]
            self.EogNum = channel_num[1]

    def get_channel_info(self):
        if self.dataset == "ISRUC":
            return [6, 2]

# TODO:各个参数为什么取这些值？随机的还是怎么弄出来的？为什么MLP的结构除了最后的输出不同其他都是一样的？MLP的参数又是如何生成的？
class EncoderConfig(object):
    def __init__(self):
        self.n_head = 8
        self.d_model = 512
        self.layer_num = 3
        self.drop = 0.1


class SleepMlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 5]


class FaceMlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 9]


class BCI2000MlpParam(object):
    def __init__(self):
        self.drop = 0.1
        self.first_linear = [512, 256]
        self.second_linear = [256, 128]
        self.out_linear = [128, 4]
