import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


def load_model_vietocr():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    #config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    #config['weights'] = 'weights/rec/VietOCR-best.pth'
    detector = Predictor(config)
    return detector