# coding=utf-8
import codecs
import collections
import json
import os
import pickle
from functools import partial

import tensorflow as tf

from bert_base import tokenization

import numpy as np
from prepro_utils import preprocess_text, encode_ids

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

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_start_label, words_label, label_x,
                 is_weight_focus, label_x_mask, weight_focus_mask, weight_matrix_mask, token_weight, piece_list=None,
                 weight_matrix=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_start_label = is_start_label
        self.piece_list = piece_list
        self.weight_matrix = weight_matrix
        self.words_label = words_label
        self.label_x = label_x
        self.is_weight_focus = is_weight_focus
        self.label_x_mask = label_x_mask
        self.weight_focus_mask = weight_focus_mask
        self.weight_matrix_mask = weight_matrix_mask
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
    max_seq_length = max_seq_length - 2
    if label_map is None:
        print("no such label2id.pkl in {} or the param label_map is empty".format(data_dir))
        return None
    tokens = []
    label = []
    is_start_token = []
    pieces_list = []
    # weight according length
    token_weight = []
    words_label = []
    label_x = []
    is_weight_focus = []
    label_x_mask = []
    weight_focus_mask = []
    for sentence in sentences:
        if not sentence:
            continue
        words, labels = zip(*sentence)
        t, l, ist, pieces, t_weight, wl, lx, iswf, lxm, wfm = process_seq(words, labels, tokenizer, label_map["X"],
                                                                          lower)

        if len(t) > max_seq_length:
            yield tokens, label, is_start_token, pieces_list, token_weight, words_label, label_x, is_weight_focus, label_x_mask, weight_focus_mask
            tokens = []
            label = []
            is_start_token = []
            pieces_list = []
            words_label = []
            is_weight_focus = []
            token_weight = []
            label_x = []
            label_x_mask = []
            weight_focus_mask = []
            # this operation will combine multiple sentences into one sequence
            t = [t[i:i + max_seq_length] for i in range(0, len(t), max_seq_length)]
            l = [l[i:i + max_seq_length] for i in range(0, len(l), max_seq_length)]
            lx = [lx[i:i + max_seq_length] for i in range(0, len(lx), max_seq_length)]
            ist = [ist[i:i + max_seq_length] for i in range(0, len(ist), max_seq_length)]
            pieces = [pieces[i:i + max_seq_length] for i in range(0, len(pieces), max_seq_length)]
            t_weight = [t_weight[i:i + max_seq_length] for i in range(0, len(t_weight), max_seq_length)]
            iswf = [iswf[i:i + max_seq_length] for i in range(0, len(iswf), max_seq_length)]
            lxm = [lxm[i:i + max_seq_length] for i in range(0, len(lxm), max_seq_length)]
            wfm = [wfm[i:i + max_seq_length] for i in range(0, len(wfm), max_seq_length)]
            start_postion = 0
            temp_list = []
            for one_row_is in ist:
                temp_list.append(wl[start_postion:start_postion + sum(one_row_is)])
                start_postion += sum(one_row_is)
            z = zip(t, l, ist, pieces, t_weight, temp_list, lx, iswf, lxm, wfm)
            for i in z:
                yield i
            continue
        if len(t) + len(tokens) > max_seq_length:
            yield tokens, label, is_start_token, pieces_list, token_weight, words_label, label_x, is_weight_focus, label_x_mask, weight_focus_mask
            tokens = t
            label = l
            is_start_token = ist
            pieces_list = pieces
            token_weight = t_weight
            words_label = wl
            label_x = lx
            is_weight_focus = iswf
            label_x_mask = lxm
            weight_focus_mask = wfm
        else:
            tokens.extend(t)
            label.extend(l)
            is_start_token.extend(ist)
            pieces_list.extend(pieces)
            token_weight.extend(t_weight)
            words_label.extend(wl)
            label_x.extend(lx)
            is_weight_focus.extend(iswf)
            label_x_mask.extend(lxm)
            weight_focus_mask.extend(wfm)
    if tokens:
        yield tokens, label, is_start_token, pieces_list, token_weight, words_label, label_x, is_weight_focus, label_x_mask, weight_focus_mask


def convert_sentences_to_simple_features(sentences, data_dir, tokenizer, max_seq_length, label_map=None, lower=True):
    if label_map is None:
        label_map = {}
    sep = os.sep if os.sep in data_dir else "/"
    for sentence in sentences:
        if not sentence:
            continue
        words, labels = zip(*sentence)
        tokens, label, is_start_token, pieces_list, token_weight, words_label, label_x, is_weight_focus, label_x_mask, weight_focus_mask = process_seq(
            words, labels, tokenizer, label_map["X"],
            lower)
        # drop if token is longer than max_seq_length
        if tokens:
            if len(tokens) > max_seq_length - 1:
                tokens = tokens[0: max_seq_length - 2]
                label = label[0:max_seq_length - 2]
                is_start_token = is_start_token[0:max_seq_length - 2]
                pieces_list = pieces_list[0:max_seq_length - 2]
                token_weight = token_weight[0:max_seq_length - 2]
                words_label = words_label[0: max_seq_length - 2]
                label_x = label_x[0: max_seq_length - 2]
                is_weight_focus = is_weight_focus[0: max_seq_length - 2]
                label_x_mask = label_x_mask[0: max_seq_length - 2]
                weight_focus_mask = weight_focus_mask[0: max_seq_length - 2]
            yield tokens, label, is_start_token, pieces_list, token_weight, words_label, label_x, is_weight_focus, label_x_mask, weight_focus_mask


def process_seq(words, labels, tokenizer, label_x, lower=True):
    assert len(words) == len(labels)
    tokens = []
    label = []
    is_start_token = []
    pieces_list = []
    token_weight = []
    words_label = []
    label_x_list = []
    is_weight_focus = []
    label_x_mask = []
    weight_focus_mask = []
    for i in range(len(words)):
        word = tokenization.convert_to_unicode(words[i])
        pieces = tokenizer.tokenize(word)
        t = tokenizer.convert_tokens_to_ids(pieces)
        tokens.extend(t)
        pieces_list.extend(pieces)
        label.extend([int(labels[i])] * len(t))
        words_label.append(int(labels[i]))
        is_start_token.append(1)
        label_x_list.append(int(labels[i]))
        label_x_mask.append(1)
        for _ in range(len(t) - 1):
            is_start_token.append(0)
            label_x_list.append(label_x)
            label_x_mask.append(0)
        all_word_length = float(len(words[i]))
        focus_index = 0
        focus_word_length = 0
        piece_index = 0
        for piece in pieces:
            piece_length = len(piece.replace("##", ""))
            if focus_word_length < piece_length:
                focus_word_length = piece_length
                focus_index = piece_index
            token_weight.append(float(piece_length) / all_word_length)
            piece_index += 1
        temp_list = [label_x] * len(pieces)
        temp_list[focus_index] = labels[i]
        is_weight_focus.extend(temp_list)
        temp_list = [0] * len(pieces)
        temp_list[focus_index] = 1
        weight_focus_mask.extend(temp_list)
    return tokens, label, is_start_token, pieces_list, token_weight, words_label, label_x_list, is_weight_focus, label_x_mask, weight_focus_mask


def single_example(ex_index, tokens, labels, is_start_token, pieces_list, token_weight, words_label_list, label_x_inp,
                   is_weight_focus, lxm, wfm, label_map, max_seq_length, mode, data_dir, biobert=False):
    weight_matrix = []
    tokens_length = len(tokens)
    input_ids = []
    input_mask = []
    segment_ids = []
    label_ids = []
    is_start_label = []
    pieces = []
    t_weight = []
    words_label = []
    weight_focus = []
    label_x = []
    label_x_mask = []
    weight_focus_mask = []
    weight_matrix_mask = []
    token_weight_list = []
    # add [CLS] to the start position
    input_ids.append(CLS_ID)
    label_ids.append(label_map["O"])
    is_start_label.append(1)
    pieces.append("[CLS]")
    words_label.append(label_map["O"])
    weight_focus.append(label_map["O"])
    label_x.append(label_map["O"])
    label_x_mask.append(1)
    weight_focus_mask.append(1)
    t_weight.append(1.0)
    t_weight.extend([0.0] * (max_seq_length - 1))
    weight_matrix.append(t_weight)
    token_weight_list.append(1)

    input_ids.extend(tokens)
    input_mask.extend([1] * (tokens_length + 2))
    segment_ids.extend([SEG_ID_A] * (tokens_length + 2))
    label_ids.extend(labels)
    words_label.extend(words_label_list)
    is_start_label.extend(is_start_token)
    weight_focus.extend(is_weight_focus)
    label_x.extend(label_x_inp)
    token_weight_list.extend(token_weight)
    # for show res
    label_x_mask.extend(lxm)
    weight_focus_mask.extend(wfm)
    pieces.extend(pieces_list)

    i = 0
    while i < len(tokens):
        t_weight = [0.0]
        for _ in range(0, i):
            t_weight.append(0.0)
        if token_weight[i] == 1.0:
            t_weight.append(1.0)
            t_weight.extend([0.0] * (max_seq_length - i - 2))
            weight_matrix.append(t_weight)
            i += 1
        else:
            t_weight.append(token_weight[i])
            j = i
            for nist in is_start_token[i + 1:]:
                if nist:
                    t_weight.extend([0.0] * (max_seq_length - j - 2))
                    weight_matrix.append(t_weight)
                    break
                j = j + 1
                t_weight.append(token_weight[j])
            i = j + 1

    # add [SEP] to the end position
    input_ids.append(SEP_ID)
    label_ids.append(label_map["O"])
    words_label.append(label_map["O"])
    is_start_label.append(1)
    pieces.append("[SEP]")
    weight_focus.append(label_map["O"])
    label_x.append(label_map["O"])
    label_x_mask.append(1)
    weight_focus_mask.append(1)
    t_weight = [0.0] * (len(tokens) + 1)
    t_weight.append(1.0)
    t_weight.extend([0.0] * (max_seq_length - len(t_weight)))
    token_weight_list.append(1)
    # weight_matrix.append(t_weight)
    # t_weight = [0.0] * (len(tokens) + 2)
    # t_weight.append(1.0)
    # t_weight.extend([0.0] * (max_seq_length - len(t_weight)))
    # weight_matrix.append(t_weight)

    for _ in range(max_seq_length - tokens_length - 2):
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(SEG_ID_A)
        label_ids.append(label_map["O"])
        is_start_label.append(0)
        pieces.append("[PAD]")
        weight_focus.append(label_map["O"])
        label_x.append(label_map["O"])
        label_x_mask.append(0)
        weight_focus_mask.append(0)
        token_weight_list.append(1)
    words_label.extend([label_map["O"]] * (max_seq_length - len(words_label)))
    for _ in range(len(weight_matrix), max_seq_length):
        weight_matrix.append([0.0] * max_seq_length)
    # add weight_matrix_mask
    weight_matrix_mask.extend([1] * sum(is_start_label))
    weight_matrix_mask.extend([0] * (max_seq_length - sum(is_start_label)))

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(is_start_label) == max_seq_length
    assert len(weight_matrix) == max_seq_length
    assert len(words_label) == max_seq_length
    assert len(weight_focus) == max_seq_length
    assert len(label_x) == max_seq_length
    assert len(label_x_mask) == max_seq_length
    assert len(weight_focus_mask) == max_seq_length
    assert len(weight_matrix_mask) == max_seq_length
    assert len(token_weight_list) == max_seq_length
    # print some example
    if ex_index < 1:
        tf.logging.info("*** Example {}***".format(mode))
        tf.logging.info("pieces: %s" % " ".join(piece for piece in pieces))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("is_start_label %s" % " ".join([str(x) for x in is_start_label]))
        tf.logging.info("words_label %s" % " ".join([str(x) for x in words_label]))
        tf.logging.info("weight_focus %s" % " ".join([str(x) for x in weight_focus]))
        tf.logging.info("label_x %s" % " ".join([str(x) for x in label_x]))
        tf.logging.info("label_x_mask %s" % " ".join([str(x) for x in label_x_mask]))
        tf.logging.info("weight_focus_mask %s" % " ".join([str(x) for x in weight_focus_mask]))
        tf.logging.info("weight_matrix_mask %s" % " ".join([str(x) for x in weight_matrix_mask]))
        tf.logging.info("token_weight %s" % " ".join([str(x) for x in token_weight_list]))
        example = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "label_ids": label_ids,
            "is_start_label": is_start_label,
            "pieces": pieces,
            "weight_matrix": weight_matrix,
            "words_label": words_label,
            "weight_focus": weight_focus,
            "label_x_mask": label_x_mask,
            "weight_focus_mask": weight_focus_mask,
            "weight_matrix_mask": weight_matrix_mask,
            "token_weight": token_weight_list
        }
        if not os.path.exists("/".join([data_dir, "{}".format("tfrecord_data")])):
            os.makedirs("/".join([data_dir, "{}".format("tfrecord_data")]))
        with open("/".join([data_dir, "{}/{}_example.json".format("tfrecord_data", mode)]), "w") as f:
            f.write(json.dumps(example, indent=4))
    # save token original sample in test data
    if mode == "test":
        return InputFeatures(input_ids, input_mask, segment_ids, label_ids, is_start_label, words_label, label_x,
                             weight_focus, label_x_mask, weight_focus_mask, weight_matrix_mask, token_weight_list,
                             piece_list=pieces, weight_matrix=weight_matrix)
    return InputFeatures(input_ids, input_mask, segment_ids, label_ids, is_start_label, words_label, label_x,
                         weight_focus, label_x_mask, weight_focus_mask, weight_matrix_mask, token_weight_list,
                         weight_matrix=weight_matrix)


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
        features["weight_matrix"] = create_float_feature(np.array(example.weight_matrix).flatten())
        features["words_label"] = create_int_feature(example.words_label)
        features["label_x"] = create_int_feature(example.label_x)
        features["is_weight_focus"] = create_int_feature(example.is_weight_focus)
        features["label_x_mask"] = create_int_feature(example.label_x_mask)
        features["weight_focus_mask"] = create_int_feature(example.weight_focus_mask)
        features["weight_matrix_mask"] = create_int_feature(example.weight_matrix_mask)
        # features["token_weight"] = create_float_feature(example.token_weight)
        if task == "test":
            features["piece_list"] = create_string_feature(example.piece_list)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    tf.logging.info("write finish!")
    writer.close()


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_data(cls, input_file, split_char="\t", label_map=None):
        """Reads a BIO data. convert to [original token1,label_id1 ...] """
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
                        line[1] = label_map["O"]
                    if sentences:
                        sentences[-1].append([line[0], line[1]])
                    else:
                        sentences.append([[line[0], line[1]]])
                else:
                    sentences.append([])
            return sentences

    def _get_label_map(self, input_file, split_char="\t", base_dir=None):
        label_map = {}
        if os.path.exists(self.sep.join([base_dir, 'label2id.pkl'])):
            with open(os.path.join(base_dir, 'label2id.pkl'), 'r') as f:
                for line in f.readlines():
                    if line:
                        lines = line.strip().split(" ")
                        if len(lines) != 2:
                            continue
                        label_map[lines[0]] = int(lines[1].replace("\n", ""))
        else:
            with open(input_file, 'r') as f:
                lines = f.readlines()
                label_map["O"] = 0
                for line in lines:
                    line = line.strip()
                    if line:
                        line = line.split(split_char)
                        if line[1] not in label_map.keys():
                            label_map[line[1]] = len(label_map)
                label_map["X"] = len(label_map)
            with open(self.sep.join([base_dir, 'label2id.pkl']), "w") as wf:
                write_str = ""
                for key in label_map.keys():
                    write_str += "{} {}\n".format(key, label_map[key])
                wf.write(write_str)
        return label_map


class NerProcessor(DataProcessor):
    def __init__(self, vocab, data_dir, max_seq_length, FLAGS, processor_data_name="", lower=True):
        self.labels = set()
        self.vocab = vocab
        self.lower = lower
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.processor_data_name = processor_data_name
        self.sep = os.sep if os.sep in self.data_dir else "/"
        self.label_map = self._get_label_map(self.sep.join([self.data_dir, "train.tsv"]), base_dir=self.data_dir)
        self.FLAGS = FLAGS
        self.get_train_examples()
        self.get_dev_examples()
        self.get_test_examples()
        self.classes = len(self.label_map) - 1

    def get_train_examples(self):
        if not tf.io.gfile.exists(self.get_record_path("train", self.FLAGS)):
            self._create_tfrecord(
                self._read_data(self.sep.join([self.data_dir, "train.tsv"]), label_map=self.label_map), "train",
                self.FLAGS)

    def get_dev_examples(self):
        if not tf.io.gfile.exists(self.get_record_path("devel", self.FLAGS)):
            return self._create_tfrecord(
                self._read_data(self.sep.join([self.data_dir, "devel.tsv"]), label_map=self.label_map), "devel",
                self.FLAGS)

    def get_test_examples(self):
        if not tf.io.gfile.exists(self.get_record_path("test", self.FLAGS)):
            return self._create_tfrecord(
                self._read_data(self.sep.join([self.data_dir, "test.tsv"]), label_map=self.label_map), "test",
                self.FLAGS)

    def get_record_path(self, mode, FLAGS):
        if FLAGS.data_mode == "combine":
            return self.sep.join([self.data_dir, "tfrecord_data",
                                  "{}_{}_{}.tfrecord".format(self.processor_data_name, mode, self.max_seq_length)])
        elif FLAGS.data_mode == "simple":
            return self.sep.join([self.data_dir, "simple_tfrecord_data",
                                  "{}_{}_{}.tfrecord".format(self.processor_data_name, mode, self.max_seq_length)])

    def get_path(self, mode):
        return self.sep.join([self.data_dir, "tfrecord_data",
                              "{}_{}_{}.tfrecord".format(self.processor_data_name, mode, self.max_seq_length)])

    def get_train_data(self):
        return self.get_path("train")

    def get_dev_data(self):
        return self.get_path("devel")

    def get_test_data(self):
        return self.get_path("test")

    def _create_tfrecord(self, sentences, task, FLAGS):
        tf.logging.set_verbosity(tf.logging.INFO)
        sep = os.sep if os.sep in self.data_dir else "/"
        top_token_num = None
        if os.path.exists(sep.join([self.data_dir, "token_num.json"])):
            with open(sep.join([self.data_dir, "token_num.json"]), "r") as f:
                top_token_num = json.loads(f.read())
        else:
            top_token_num = {}
        examples = []
        i = 0
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab, do_lower_case=self.lower)
        # for data in convert_sentences_to_simple_features(sentences, self.data_dir, tokenizer, self.max_seq_length, self.label_map):
        features = convert_sentences_to_features(sentences, self.data_dir, tokenizer, self.max_seq_length,
                                                 self.label_map) if FLAGS.data_mode == "combine" else convert_sentences_to_simple_features(
            sentences, self.data_dir, tokenizer, self.max_seq_length, self.label_map)
        for data in features:
            examples.append(
                single_example(i, *data, label_map=self.label_map, max_seq_length=self.max_seq_length, mode=task,
                               data_dir=self.data_dir, biobert=FLAGS.biobert))
            get_token_num(top_token_num, data[2])
            i += 1
        if not os.path.exists(sep.join([self.data_dir, "tfrecord_data"])):
            os.mkdir(sep.join([self.data_dir, "tfrecord_data"]))
        output_file = sep.join(
            [self.data_dir, "tfrecord_data",
             "{}_{}_{}.tfrecord".format(self.processor_data_name, task, self.max_seq_length)])
        file_based_convert_examples_to_features(examples, output_file, task)
        if not os.path.exists(sep.join([self.data_dir, "token_num.json"])):
            with open(sep.join([self.data_dir, "token_num.json"]), "w") as f:
                f.write(json.dumps(top_token_num))
        return len(examples)

    def get_train_step(self):
        c = 0
        for record in tf.python_io.tf_record_iterator(self.get_path("train")):
            c += 1
        for record in tf.python_io.tf_record_iterator(self.get_path("devel")):
            c += 1
        print("record_num {}".format(str(c)))
        # tf.logging.info("train will end at step {}".format(str(c)))
        return int(c / self.FLAGS.train_batch_size * self.FLAGS.train_epoch)


def get_token_num(top_token_num, is_start_token):
    if top_token_num is None:
        top_token_num = {}
    i = 0
    while i < len(is_start_token):
        if is_start_token[i] == 1:
            word_length = 1
            j = 0
            for nist in is_start_token[i + 1:]:
                if nist:
                    break
                word_length += 1
                j = j + 1
            if word_length not in top_token_num:
                top_token_num[word_length] = 1
            else:
                top_token_num[word_length] += 1
            i = i + j + 1
