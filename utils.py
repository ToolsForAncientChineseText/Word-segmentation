import os
from produce_tf import produce_tfrecord


def read_txt_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        return file.read().split('\n')


def save_txt_file(filename, sents):
    with open(filename, 'w', encoding="utf-8") as file:
        for idx, sent in enumerate(sents):
            if idx != len(sents) - 1:
                file.write(sent + '\n')
            else:
                file.write(sent)


def to_label(line):
    words = line.split(' ')
    label = []
    for word in words:
        label.append("[BOS]")
        rem = len(word) - 1
        for _ in range(rem):
            label.append("[IOS]")
    return "".join(words), " ".join(label)


def to_seg_data(OUTPUT_FILE, out_filename="./data/temp_pred_will_be_del_when_pred_finished.txt"):
    if out_filename is None:
        out_filename = OUTPUT_FILE
    seg_lines = []
    with open(OUTPUT_FILE, 'r', encoding="utf-8") as file:
        lines = file.read().split('\n')
        for line in lines:
            line, label = to_label(line)
            if len(line) > 22:
                continue
            seg_lines.append(line + '|' + label)
    with open(out_filename, 'w', encoding="utf-8") as file:
        for i, seg in enumerate(seg_lines):
            if i != len(seg_lines) - 1:
                seg = seg + '\n'
            file.write(seg)


def labeled_txt_to_tf(filename):
    to_seg_data(filename)
    produce_tfrecord()


def label2text(text, label):
    res = []
    label = label.split(' ')[1:-1]
    idx = 0
    for i in range(len(label)):
        if label[i] != "[BOS]" and label[i] != "[IOS]":
            continue
        if idx >= len(text):
            break
        if label[i] == "[BOS]" and res != []:
            res.append(' ')
        res.append(text[idx])
        idx += 1
    return "".join(res)


def from_label_to_text(text_file, label_file, output_file):
    with open(text_file, 'r', encoding="utf-8") as file:
        gt = file.read().split('\n')
    with open(label_file, 'r', encoding="utf-8") as file:
        pred = file.read().split('\n')
    with open(output_file, 'w', encoding="utf-8") as file:
        for i in range(len(gt)):
            if gt[i] == "":
                continue
            text = "".join(gt[i].split(' '))
            if len(text) == 0:
                continue
            pred_label = pred[i]
            prediction = label2text(text, pred_label)
            if i == len(gt) - 1:
                file.write(prediction)
            else:
                file.write(prediction + '\n')


def fix():
    with open("data/temp_pred_will_be_del.txt", 'r', encoding="utf-8") as file:
        lines = file.read().split('\n')
    res = []
    for line in lines:
        line = line.split(' ')[1:-1]
        has_bos = False
        temp = ["[CLS]"]
        for tag in line:
            if tag == "[SEP]" or tag == "[CLS]":
                temp.append("[BOS]")
            elif tag == "[IOS]" and has_bos is False:
                temp.append("[BOS]")
            else:
                if tag == "[BOS]":
                    has_bos = True
                temp.append(tag)
        temp.append("[SEP]")
        res.append(temp)

    with open("data/temp_pred_will_be_del.txt", 'w', encoding="utf-8") as file:
        for line in res:
            file.write(" ".join(line) + '\n')
