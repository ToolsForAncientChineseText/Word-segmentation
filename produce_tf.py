#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:WeiYi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tokenization
import tensorflow as tf
from tensorflow.python.ops import math_ops
import pickle

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name", "to_tfrecord", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", "./data/temp_pred_will_be_del_when_pred_finished.txt",
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", "./data/pred.tf_record",
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_bool(
    "req_do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "data_max_seq_length", 24,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool("use_tpu_", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("used_vocab_file", "./vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master_", None, "[Optional] TensorFlow master_ URL.")
flags.DEFINE_integer(
    "num_tpu_cores_", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = []
            for line in f:
                content = line.split('|')
                sent = content[0].strip()
                label = content[1].replace('\n', '')
                lines.append([sent, label])
            return lines


class TfrecordProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(data_dir), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["[BOS]", "[IOS]", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    with open('./label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    total_written = 0
    for (ex_index, example) in enumerate(examples):
        try:
            feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        except:
            continue
        total_written += 1
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    tf.logging.info("Wrote %d total instances", total_written)

def produce_tfrecord():
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "to_tfrecord": TfrecordProcessor
    }

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.used_vocab_file, do_lower_case=FLAGS.req_do_lower_case)

    train_file = os.path.join(FLAGS.output_dir)
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    filed_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.data_max_seq_length, tokenizer, train_file)
