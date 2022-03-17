import os
import re
import logging
import binascii
from argparse import ArgumentParser

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


def clean_csv(input_path, output_path):
    flag = False
    writer = open(output_path, "w")

    for idx, line in enumerate(open(input_path)):
        line = re.sub(" +", " ", line)
        if idx == 0:
            writer.write(line.replace("\n", " "))
        else:
            if not line.startswith('"') or line.startswith('""'):
                flag = True
                writer.write(line.replace("\n", " "))
            else:
                if flag:
                    flag = False
                writer.write("\n")
                writer.write(line.replace("\n", " "))
    writer.close()


def bigram_generation(packet_string, packet_len=64, flag=True):
    result = ""
    sentence = cut(packet_string, 1)
    token_count = 0
    for sub_string_index in range(len(sentence)):
        if sub_string_index != (len(sentence) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = (
                    sentence[sub_string_index] + sentence[sub_string_index + 1]
                )
        else:
            break
        result += merge_word_bigram
        result += " "
    if flag == True:
        result = result.rstrip()

    return result


def cut(obj, sec):
    result = [obj[i : i + sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except Exception as e:
        remanent_count = 0
        print(1)
    if remanent_count == 0:
        pass
    else:
        result = [
            obj[i : i + sec + remanent_count]
            for i in range(0, len(obj), sec + remanent_count)
        ]
    return result


def raw2body(x):
    if "|||" in x:
        body = x.split("|||")[20]
    elif '"|"' in x:
        body = x.split("|")[22]
        body = body[1:]
    elif '","' in x:
        body = x.split('","')[22]

    body = body.replace("\\r\\n", " ")
    return body


def body2hex(body, packet_len=128):
    data = binascii.hexlify(bytes(body, encoding="utf-8"))
    packet_string = data.decode()
    data_string = bigram_generation(packet_string, packet_len=packet_len)
    return data_string


def get_logger(level=logging.INFO):
    LOG_FORMAT = "[%(asctime)-10s] (line: %(lineno)d) %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger


if __name__ == "__main__":
    logger = get_logger()

    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--min_count", type=int, default=20)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser_args = parser.parse_args()

    if parser_args.clean:
        clean_csv("ai_data.csv", "ai_data_clean.csv")

    df = pd.read_csv(parser_args.input_path)
    logger.info(f"{df.signature.unique()}")
    logger.info(f"Label count : {len(df.signature.unique())}")

    df["_raw_body"] = df["_raw"].apply(lambda x: raw2body(x))

    remove_group_id = set()
    for group, group_df in df.groupby("signature"):
        if len(group_df) < parser_args.min_count:
            remove_group_id.add(group)

    df = df[~df["signature"].isin(remove_group_id)]
    df = df.reset_index()

    logger.info(f"{df.signature.unique()}")
    logger.info(f"Label count : {len(df.signature.unique())}")

    label_encoder = LabelEncoder()
    label_encoder.fit(df.signature)
    df.signature = label_encoder.transform(df.signature)

    if parser_args.split:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)

        for train_index, test_index in split.split(
            df[["_raw_body"]], df[["signature"]]
        ):
            x_payload_train, y_train = (
                df["_raw_body"][train_index],
                df["signature"][train_index],
            )
            x_payload_test, y_test = (
                df["_raw_body"][test_index],
                df["signature"][test_index],
            )

        if not os.path.exists(parser_args.output_dir):
            os.makedirs(parser_args.output_dir)

        raw_train_wrtier = open(
            os.path.join(parser_args.output_dir, "train_dataset_raw.tsv"), "w"
        )
        raw_test_writer = open(
            os.path.join(parser_args.output_dir, "test_dataset_raw.tsv"), "w"
        )

        train_wrtier = open(
            os.path.join(parser_args.output_dir, "train_dataset.tsv"), "w"
        )
        test_writer = open(
            os.path.join(parser_args.output_dir, "test_dataset.tsv"), "w"
        )

        raw_train_wrtier.write(f"label\ttext_a\n")
        train_wrtier.write(f"label\ttext_a\n")
        for x, y in zip(x_payload_train, y_train):
            raw_train_wrtier.write(f"{y}\t{x}\n")
            train_wrtier.write(f"{y}\t{body2hex(x)}\n")

        raw_test_writer.write(f"label\ttext_a\n")
        test_writer.write(f"label\ttext_a\n")
        for x, y in zip(x_payload_test, y_test):
            raw_test_writer.write(f"{y}\t{x}\n")
            test_writer.write(f"{y}\t{body2hex(x)}\n")

        raw_train_wrtier.close()
        raw_test_writer.close()

        train_wrtier.close()
        test_writer.close()
    else:
        wrtier = open(os.path.join(parser_args.output_dir, "dataset.tsv"), "w")
        raw_wrtier = open(os.path.join(parser_args.output_dir, "dataset_raw.tsv"), "w")

        wrtier.write(f"label\ttext_a\n")
        raw_wrtier.write(f"label\ttext_a\n")
        for x, y in zip(df["_raw_body"], df["signature"]):
            wrtier.write(f"{y}\t{body2hex(x)}\n")
            raw_wrtier.write(f"{y}\t{x}\n")

        wrtier.close()
        raw_wrtier.close()

    logger.info(
        f"finish generating dataset.\n please check in {parser_args.output_dir}"
    )
