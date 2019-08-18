from BERT_PRED import prediction


if __name__ == '__main__':
    prediction_filename = input("分词文件名：")
    output_filename = input("输出文件名：")
    prediction(prediction_filename, output_filename)
