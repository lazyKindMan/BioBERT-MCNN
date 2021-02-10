import codecs
import collections
import json
import os
import pickle
from functools import partial

import tensorflow as tf

from bert_base import tokenization

from ner_util.logutil import set_logger
from prepro_utils import preprocess_text, encode_ids
import sentencepiece as spm

logger = set_logger('NER Training')

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

special_symbols = {
    "[unk]": 0,
    "[s]": 1,
    "[/s]": 2,
    "[cls]": 3,
    "[sep]": 4,
    "[pad]": 5,
    "[mask]": 6,
    "[eod]": 7,
    "[eop]": 8,
}

CLS_ID = special_symbols["[cls]"]
SEP_ID = special_symbols["[sep]"]
MASK_ID = special_symbols["[mask]"]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None, is_start_token=None, pieces=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
        self.pieces = pieces
        self.is_start_token = is_start_token


class InputFeatures(object):
    """A single set of features of data for ner task."""
    """Constructor for InputFeatures.

        Args:
          input_ids: int32 Tensor of shape [seq_length]. Already been converted into WordPiece token ids
          input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
          segment_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
          label_id: int32 Tensor of shape [batch_size, seq_length]. for recording the tag corresponding to token 
        """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_start_label, token_weight, piece_list=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_start_label = is_start_label
        self.piece_list = piece_list
        self.token_weight = token_weight


def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_sentences_to_features(sentences, data_dir, tokenizer, max_seq_length, label_map=None, lower=True):
    """
    convert sentences to InputExample class
    :param sentences: all sequence input to the model
    :param data_dir: directory path for saving  data
    :param sp sentencepiece model
    :param max_seq_length the max sequence for bert model
    :param lower: if ignore the case
    :param label_map: label_map2id if empty it will read label2id.pkl file to load which is in the data_dir
    :return: for list ([tokens] [labels] [is_start_token] [pieces_list])
    """
    # compatible for windows
    if label_map is None:
        label_map = {}
    sep = os.sep if os.sep in data_dir else "/"
    # three are remained for the mark [CLS] and two [SEP] in the end
    max_seq_length = max_seq_length - 3
    # read label mapping to id
    label_map_path = sep.join([data_dir, "label2id.pkl"])
    if not label_map and os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            for line in f.readlines():
                if line:
                    lines = line.strip().split(" ")
                    if len(lines) != 2:
                        continue
                    label_map[lines[0]] = int(lines[1].replace("\n", ""))
    elif not os.path.exists(label_map_path):
        print("no such label2id.pkl in {} or the param label_map is empty".format(data_dir))
        return None
    tokens = []
    label = []
    is_start_token = []
    pieces_list = []
    # weight according length
    token_weight = []
    for sentence in sentences:
        if not sentence:
            continue
        words, labels = zip(*sentence)
        t, l, ist, pieces, t_weight = process_seq(words, labels, tokenizer, lower)

        if len(t) > max_seq_length:
            yield tokens, label, is_start_token, pieces_list,token_weight, label_map
            tokens = []
            label = []
            is_start_token = []
            pieces_list = []
            # this operation will combine multiple sentences into one sequence
            t = [t[i:i + max_seq_length] for i in range(0, len(t), max_seq_length)]
            l = [l[i:i + max_seq_length] for i in range(0, len(l), max_seq_length)]
            ist = [ist[i:i + max_seq_length] for i in range(0, len(ist), max_seq_length)]
            pieces = [pieces[i:i + max_seq_length] for i in range(0, len(pieces), max_seq_length)]
            token_weight = [t_weight[i:i + max_seq_length] for i in range(0, len(t_weight), max_seq_length)]
            z = zip(t, l, ist, pieces)
            for i in z:
                yield i
                continue
        if len(t) + len(tokens) > max_seq_length:
            yield tokens, label, is_start_token, pieces_list, token_weight, label_map
            tokens = t
            label = l
            is_start_token = ist
            pieces_list = pieces
            token_weight = t_weight
        else:
            tokens.extend(t)
            label.extend(l)
            is_start_token.extend(ist)
            pieces_list.extend(pieces)
            token_weight.extend(t_weight)
    if tokens:
        yield tokens, label, is_start_token, pieces_list, token_weight, label_map


def process_seq(words, labels, tokenizer, lower=True):
    assert len(words) == len(labels)
    tokens = []
    label = []
    is_start_token = []
    pieces_list = []
    token_weight = []
    for i in range(len(words)):
        word = tokenization.convert_to_unicode(words[i])
        pieces = tokenizer.tokenize(word)
        t = tokenizer.convert_tokens_to_ids(pieces)
        tokens.extend(t)
        pieces_list.extend(pieces)
        label.extend([int(labels[i])] * len(t))
        is_start_token.append(1)
        for _ in range(len(t) - 1):
            is_start_token.append(0)
        all_word_length = float(len(words[i]))
        for piece in pieces:
            token_weight.append(float(len(piece.replace("##", ""))) / all_word_length)
    return tokens, label, is_start_token, pieces_list, token_weight


def single_example(ex_index, tokens, labels, is_start_token, pieces_list, token_weight, label_map, max_seq_length, mode, data_dir):
    tokens_length = len(tokens)
    input_ids = []
    input_mask = []
    segment_ids = []
    label_ids = []
    is_start_label = []
    pieces = []
    t_weight = []
    # add [CLS] to the start position
    input_ids.append(CLS_ID)
    label_ids.append(label_map["O"])
    is_start_label.append(1)
    pieces.append("[CLS]")
    t_weight.append(1.0)

    input_ids.extend(tokens)
    input_mask.extend([1] * (tokens_length + 3))
    segment_ids.extend([SEG_ID_A] * (tokens_length + 3))
    label_ids.extend(labels)
    is_start_label.extend(is_start_token)
    t_weight.extend(token_weight)
    # for show res
    pieces.extend(pieces_list)

    # add [SEP] to the end position
    input_ids.extend([SEP_ID, SEP_ID])
    label_ids.extend([label_map["O"]] * 2)
    is_start_label.extend([1, 1])
    pieces.extend(["[SEP]", "[SEP]"])
    t_weight.extend([1.0, 1.0])

    for _ in range(max_seq_length - tokens_length - 3):
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(SEG_ID_PAD)
        label_ids.append(label_map["O"])
        is_start_label.append(0)
        pieces.append("*")
        t_weight.append(1.0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(is_start_label) == max_seq_length
    assert len(t_weight) == max_seq_length
    # print some example
    if ex_index < 1:
        logger.info("*** Example {}***".format(mode))
        logger.info("pieces: %s" % " ".join(piece for piece in pieces))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        logger.info("is_start_label %s" % " ".join([str(x) for x in is_start_label]))
        logger.info("token_weight %s" % " ".join([str(x) for x in t_weight]))
        example = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "label_ids": label_ids,
            "is_start_label": is_start_label,
            "token_weight": t_weight,
            "pieces": pieces
        }
        if not os.path.exists("/".join([data_dir, "{}".format("tfrecord_data")])):
            os.makedirs("/".join([data_dir, "{}".format("tfrecord_data")]))
        with open("/".join([data_dir, "{}/{}_example.json".format("tfrecord_data", mode)]), "w") as f:
            f.write(json.dumps(example, indent=4))
    # save token original sample in test data
    if mode == "test":
        return InputFeatures(input_ids, input_mask, segment_ids, label_ids, is_start_label, t_weight, pieces)
    return InputFeatures(input_ids, input_mask, segment_ids, label_ids, is_start_label, t_weight)


def file_based_convert_examples_to_features(examples, output_file, task):
    tf.logging.info("Start writing tfrecord %s.", output_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    tf.logging.info("total %d examples", len(examples))
    for ex_index, example in enumerate(examples):
        if ex_index % 100 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        def create_string_feature(values):
            byte_values = [bytes(s, encoding='utf-8') for s in values]
            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(byte_values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(example.input_ids)
        features["input_mask"] = create_int_feature(example.input_mask)
        features["segment_ids"] = create_int_feature(example.segment_ids)
        features["label_ids"] = create_int_feature(example.label_ids)
        features["is_start_label"] = create_int_feature(example.is_start_label)
        features["token_weight"] = create_float_feature(example.token_weight)
        if task == "test":
            features["piece_list"] = create_string_feature(example.piece_list)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    tf.logging.info("write finish!")
    writer.close()


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_data(cls, input_file, split_char="\t", label_map=None, base_dir=None):
        """Reads a BIO data. convert to [original token1,label_id1 ...] """
        if not label_map:
            label_map = {}
        with open(input_file, 'r') as f:
            lines = f.readlines()
            sentences = []
            for line in lines:
                line = line.strip()
                if line:
                    line = line.split(split_char)
                    if line[1] in label_map.keys():
                        line[1] = label_map[line[1]]
                    else:
                        if len(label_map.keys()) == 0:
                            label_map[line[1]] = 0
                            line[1] = 0
                        else:
                            label_map[line[1]] = max(label_map.values()) + 1
                            line[1] = label_map[line[1]]
                    if sentences:
                        sentences[-1].append([line[0], line[1]])
                    else:
                        sentences.append([[line[0], line[1]]])
                else:
                    sentences.append([])
            # create label set Vocabulary
            if base_dir is None:
                base_dir = ""
                path_sep = os.sep if os.sep in input_file else "/"
                print(input_file.split(path_sep))
                for p in input_file.split(path_sep)[:-1]:
                    base_dir = os.path.join(base_dir, p)
            # 1表示从1开始对label进行index化
            # 保存label->index 的map
            if not os.path.exists(os.path.join(base_dir, 'label2id.pkl')):
                label_map["X"] = max(label_map.values()) + 1
                with open(os.path.join(base_dir, 'label2id.pkl'), 'w') as w:
                    for key, val in label_map.items():
                        w.write("{} {}\n".format(key, val))
            return sentences


class NerProcessor(DataProcessor):
    def __init__(self, vocab, data_dir, max_seq_length, processor_data_name="", lower=True):
        self.labels = set()
        self.vocab = vocab
        self.lower = lower
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.processor_data_name = processor_data_name
        self.sep = os.sep if os.sep in self.data_dir else "/"
        self.label_map = {}


        self.get_train_examples()
        if os.path.exists(self.sep.join([data_dir, 'label2id.pkl'])):
            with open(os.path.join(data_dir, 'label2id.pkl'), 'r') as f:
                for line in f.readlines():
                    if line:
                        lines = line.strip().split(" ")
                        if len(lines) != 2:
                            continue
                        self.label_map[lines[0]] = int(lines[1].replace("\n", ""))
        self.get_dev_examples()
        self.get_test_examples()
        self.classes = len(self.label_map) - 1

    def get_train_examples(self):
        if not tf.io.gfile.exists(self.get_record_path("train")):
            self._create_tfrecord(
                self._read_data(self.sep.join([self.data_dir, "train.tsv"]), label_map=self.label_map), "train")

    def get_dev_examples(self):
        if not tf.io.gfile.exists(self.get_record_path("devel")):
            return self._create_tfrecord(
                self._read_data(self.sep.join([self.data_dir, "devel.tsv"]), label_map=self.label_map), "eval")

    def get_test_examples(self):
        if not tf.io.gfile.exists(self.get_record_path("test")):
            return self._create_tfrecord(
                self._read_data(self.sep.join([self.data_dir, "test.tsv"]), label_map=self.label_map), "test")

    def get_record_path(self, mode):
        return self.sep.join([self.data_dir, "tfrecord_data",
                              "{}_{}_{}.tfrecord".format(self.processor_data_name, mode, self.max_seq_length)])

    def get_path(self, mode):
        return self.sep.join([self.data_dir, "tfrecord_data", "{}_{}_{}.tfrecord".format(self.processor_data_name, mode,self.max_seq_length)])

    def get_train_data(self):
        return self.get_path("train")

    def get_dev_data(self):
        return self.get_path("devel")

    def get_test_data(self):
        return self.get_path("test")


    def _create_tfrecord(self, sentences, task):
        tf.logging.set_verbosity(tf.logging.INFO)
        examples = []
        i = 0
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab, do_lower_case=self.lower)
        for data in convert_sentences_to_features(sentences, self.data_dir, tokenizer, self.max_seq_length, self.label_map):
            examples.append(single_example(i, *data, max_seq_length=self.max_seq_length, mode=task, data_dir = self.data_dir))
            i += 1
        sep = os.sep if os.sep in self.data_dir else "/"
        if not os.path.exists(sep.join([self.data_dir, "tfrecord_data"])):
            os.mkdir(sep.join([self.data_dir, "tfrecord_data"]))
        output_file = sep.join(
            [self.data_dir, "tfrecord_data", "{}_{}_{}.tfrecord".format(self.processor_data_name, task, self.max_seq_length)])
        file_based_convert_examples_to_features(examples, output_file, task)
        return len(examples)

    def get_train_step(self):
        c = 0
        for record in tf.python_io.tf_record_iterator(self.get_path("train")):
            c += 1
        print("record_num {}".format(str(c)))
        return ((c * 7) // 300 + 1) * 300  
