"""
  This script provides an exmaple to wrap UER-py for classification inference.
"""
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts
from finetuning import Classifier


class ETInferDataset(Dataset):
    def __init__(self, args, path):
        self.dataset, columns = [], {}
        for line_id, line in enumerate(open(path, mode="r", encoding="utf-8")):
            if line_id == 0:
                line = line.strip().split("\t")
                for i, column_name in enumerate(line):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
                )
                seg = [1] * len(src)
            else:  # Sentence pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                )
                src_b = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
                )
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            src = torch.LongTensor(src)
            seg = torch.LongTensor(seg)

            self.dataset.append((src, seg))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def to_numpy(tensor):
    """
    If the tensor requires gradients, then detach it from the computation graph, move it to the CPU, and
    convert it to a NumPy array. Otherwise, just move it to the CPU and convert it to a NumPy array
    
    :param tensor: A PyTorch tensor
    :return: the tensor as a numpy array.
    """
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    infer_opts(parser)

    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "last"],
        default="first",
        help="Pooling type.",
    )

    parser.add_argument(
        "--labels_num", type=int, required=True, help="Number of prediction labels."
    )

    parser.add_argument(
        "--tokenizer",
        choices=["bert", "char", "space"],
        default="bert",
        help="Specify the tokenizer."
        "Original Google BERT uses bert tokenizer on Chinese corpus."
        "Char tokenizer segments sentences into characters."
        "Space tokenizer segments sentences into words according to space.",
    )

    parser.add_argument(
        "--output_logits", action="store_true", help="Write logits to output file."
    )
    parser.add_argument(
        "--output_prob", action="store_true", help="Write probabilities to output file."
    )
    parser.add_argument("--multi_gpu", action="store_true", help="Using multi GPU")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")

    parser.add_argument(
        "--num_worker", type=int, default=1, help="Number of dataloader worker."
    )

    args = parser.parse_args()

    args = load_hyperparam(args)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    if args.num_worker > 1:
        torch.multiprocessing.set_start_method("spawn")

    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    model = model.to(args.device)

    if args.multi_gpu:
        print(
            "{} GPUs are available. Let's use them.".format(torch.cuda.device_count())
        )
        model = torch.nn.DataParallel(model)

    dataset = ETInferDataset(args, args.test_path)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=True,
    )

    model.eval()

    full_start_time = time.time()
    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        if args.output_logits:
            f.write("\t" + "logits")
        if args.output_prob:
            f.write("\t" + "prob")
        f.write("\n")

        for batch in iter(loader):
            src_batch, seg_batch = batch

            with torch.no_grad():
                _, logits = model(
                    src_batch.to(args.device), seg_batch.to(args.device)
                )

            preds = to_numpy(torch.argmax(logits, dim=1))

            if args.output_prob:
                prob = nn.Softmax(dim=1)(logits)
                prob = to_numpy(prob)

            if args.output_logits:
                logits = to_numpy(logits)

            for idx, pred in enumerate(preds):
                f.write(f"{str(pred)}")

                if args.output_logits:
                    f.write("\t" + " ".join([str(v) for v in logits[idx]]))
                if args.output_prob:
                    f.write("\t" + " ".join([str(v) for v in prob[idx]]))

                f.write(f"\n")

    print(f"inference time--{time.time()-full_start_time}s seconds---")


if __name__ == "__main__":
    main()
