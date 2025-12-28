from allennlp.data.fields import *
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.nn.util import get_text_field_mask
from allennlp.data.tokenizers import Token
from allennlp.models import BasicClassifier, Model
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure, Average, Metric
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics.metric import Metric
from allennlp.nn import util

from typing import *
from overrides import overrides
import jieba
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import cv2 as cv
import os

torch.manual_seed(123)


def process_image(img, min_side=224):  # 等比例缩放与填充
    size = img.shape
    h, w = size[0], size[1]
    # 长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    # 下右填充
    top, bottom, left, right = 0, min_side-new_h, 0, min_side-new_w

    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255]) # 从图像边界向上,下,左,右扩的像素数目

    return pad_img




@DatasetReader.register("s2s_manual_reader")
class SeqReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexer: Dict[str, TokenIndexer] = None,
                 target_token_indexer: Dict[str, TokenIndexer] = None,
                 model_name: str = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._source_token_indexer = source_token_indexer
        self._target_token_indexer = target_token_indexer
        self._model_name = model_name

        #sub_dict_path = "ViGeoQA/sub_dataset_dict.pk"  # problems type
        sub_dict_path = "/kaggle/input/vigeoqa/sub_dataset_dict.pk"  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

        # self.all_points = ['切线', '垂径定理', '勾股定理', '同位角', '平行线', '三角形内角和', '三角形中位线', '平行四边形',
        #               '相似三角形', '正方形', '圆周角', '直角三角形', '距离', '邻补角', '圆心角', '圆锥的计算', '三角函数',
        #               '矩形', '旋转', '等腰三角形', '外接圆', '内错角', '菱形', '多边形', '对顶角', '三角形的外角', '角平分线',
        #               '对称', '立体图形', '三视图', '圆内接四边形', '垂直平分线', '垂线', '扇形面积', '等边三角形', '平移',
        #               '含30度角的直角三角形', '仰角', '三角形的外接圆与外心', '方向角', '坡角', '直角三角形斜边上的中线', '位似',
        #               '平行线分线段成比例', '坐标与图形性质', '圆柱的计算', '俯角', '射影定理', '黄金分割', '钟面角']

        self.all_points = [
            'Ba hình chiếu',
            'Các đoạn thẳng tỉ lệ do đường thẳng song song',
            'Các đoạn thẳng tỉ lệ tạo bởi đường thẳng song song',
            'Diện tích hình quạt',
            'Góc chúc',
            'Góc dốc',
            'Góc hạ',
            'Góc kề bù',
            'Góc ngoài của tam giác',
            'Góc ngoài tam giác',
            'Góc nâng',
            'Góc nội tiếp',
            'Góc phương hướng',
            'Góc phương vị',
            'Góc so le trong',
            'Góc trên mặt đồng hồ',
            'Góc đối đỉnh',
            'Góc đồng vị',
            'Góc ở tâm',
            'Hàm số lượng giác',
            'Hình bình hành',
            'Hình chiếu ba mặt',
            'Hình chữ nhật',
            'Hình không gian',
            'Hình khối',
            'Hình khối không gian',
            'Hình lập thể',
            'Hình thoi',
            'Hình vuông',
            'Hệ quả của định lý Ta-lét',
            'Khoảng cách',
            'Phân giác góc',
            'Phép quay',
            'Phép tính hình nón',
            'Phép tịnh tiến',
            'Phép vị tự',
            'Sự quay',
            'Tam giác cân',
            'Tam giác vuông',
            'Tam giác vuông chứa góc 30 độ',
            'Tam giác vuông có góc 30 độ',
            'Tam giác vuông có một góc 30 độ',
            'Tam giác đều',
            'Tam giác đồng dạng',
            'Tia phân giác',
            'Tia phân giác của góc',
            'Tia phân giác góc',
            'Tiếp tuyến',
            'Trung tuyến ứng với cạnh huyền của tam giác vuông',
            'Tính toán hình nón',
            'Tính toán hình trụ',
            'Tịnh tiến',
            'Tọa độ và tính chất của hình',
            'Tọa độ và tính chất hình học',
            'Tổng ba góc trong của tam giác',
            'Tổng ba góc trong tam giác',
            'Tổng các góc trong của tam giác',
            'Tổng các góc trong tam giác',
            'Tứ giác nội tiếp',
            'Tứ giác nội tiếp đường tròn',
            'Tỷ lệ vàng',
            'Vị tự',
            'Đa giác',
            'Đường phân giác',
            'Đường phân giác của góc',
            'Đường phân giác góc',
            'Đường song song',
            'Đường song song cắt đoạn thẳng thành các đoạn tỉ lệ',
            'Đường thẳng song song',
            'Đường thẳng song song chia đoạn thẳng thành các đoạn thẳng tỉ lệ',
            'Đường thẳng song song chia đoạn thẳng thành các đoạn tỉ lệ',
            'Đường thẳng song song chia đoạn thẳng tỉ lệ',
            'Đường thẳng song song cắt đoạn thẳng tỉ lệ',
            'Đường thẳng song song và các đoạn thẳng tỉ lệ',
            'Đường thẳng vuông góc',
            'Đường trung bình của tam giác',
            'Đường trung trực',
            'Đường trung tuyến ứng với cạnh huyền của tam giác vuông',
            'Đường trung tuyến ứng với cạnh huyền trong tam giác vuông',
            'Đường tròn ngoại tiếp',
            'Đường tròn ngoại tiếp và tâm đường tròn ngoại tiếp của tam giác',
            'Đường tròn ngoại tiếp và tâm đường tròn ngoại tiếp tam giác',
            'Đường vuông góc',
            'Định lí Py-ta-go',
            'Định lí Pytago',
            'Định lí đường kính vuông góc với dây cung',
            'Định lí đường kính và dây cung',
            'Định lý Pitago',
            'Định lý Py-ta-go',
            'Định lý Py-tha-go',
            'Định lý Pytago',
            'Định lý Pythagoras',
            'Định lý Pythagore',
            'Định lý Ta-lét',
            'Định lý Ta-lét về các đoạn thẳng tỉ lệ',
            'Định lý Ta-lét về đoạn thẳng tỉ lệ',
            'Định lý Talet',
            'Định lý Talet (Đường thẳng song song và các đoạn thẳng tỉ lệ)',
            'Định lý Thales',
            'Định lý về các đoạn thẳng tỉ lệ do đường thẳng song song tạo ra',
            'Định lý về đoạn thẳng tỉ lệ do đường thẳng song song tạo ra',
            'Định lý về đường thẳng song song và đoạn thẳng tỉ lệ',
            'Định lý đường kính vuông góc với dây cung',
            'Định lý đường kính và dây cung',
            'Đối xứng',
        ]

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            for sample in dataset:
                yield self.text_to_instance(sample)

    @overrides
    def text_to_instance(self, sample) -> Instance:
        fields = {}

        image = sample['image']
        image = process_image(image)
        image = image/255
        img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
        for i in range(3):
            img_rgb[i, :, :] = image
        fields['image'] = ArrayField(img_rgb)

        s_token = self._tokenizer.tokenize(' '.join(sample['token_list']))
        fields['source_tokens'] = TextField(s_token, self._source_token_indexer)
        t_token = self._tokenizer.tokenize(' '.join(sample['manual_program']))
        t_token.insert(0, Token(START_SYMBOL))
        t_token.append(Token(END_SYMBOL))
        fields['target_tokens'] = TextField(t_token, self._target_token_indexer)
        fields['source_nums'] = MetadataField(sample['numbers'])
        fields['choice_nums'] = MetadataField(sample['choice_nums'])
        fields['label'] = MetadataField(sample['label'])

        type = self.subset_dict[sample['id']]
        fields['type'] = MetadataField(type)
        fields['data_id'] = MetadataField(sample['id'])

        equ_list = []

        equ = sample['manual_program']
        equ_token = self._tokenizer.tokenize(' '.join(equ))
        equ_token.insert(0, Token(START_SYMBOL))
        equ_token.append(Token(END_SYMBOL))
        equ_token = TextField(equ_token, self._source_token_indexer)
        equ_list.append(equ_token)

        fields['equ_list'] = ListField(equ_list)
        fields['manual_program'] = MetadataField(sample['manual_program'])

        point_label = np.zeros(50, np.float32)
        exam_points = sample['formal_point']
        for point in exam_points:
            point_id = self.all_points.index(point)
            point_label[point_id] = 1
        fields['point_label'] = ArrayField(np.array(point_label))

        return Instance(fields)



