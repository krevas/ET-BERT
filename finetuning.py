"""
This script provides an exmaple to wrap UER-py for classification.
"""
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite

from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, seg, tgt=None, soft_tgt=None):
        # Embedding.
        emb_data = self.embedding(src, seg)
        # Encoder.
        for idx, each_batch_size in enumerate(range(emb_data.size(0))):
            emb = emb_data[each_batch_size]
            seg_data = seg[each_batch_size]
            output_emb = self.encoder(emb, seg_data)
            output_data = output_emb[:, :1, :]
            ### delete dim of seq_length, expand dim of batch
            cls_output = output_data.squeeze(1).unsqueeze(0)
            if idx == 0:
                output = cls_output
            else:
                output = torch.cat((output, cls_output), 0)

        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + (
                    1 - self.soft_alpha
                ) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


class ETTrainDataset(Dataset):
    def __init__(self, args, path):
        self.dataset, columns = [], {}
        for line_id, line in enumerate(open(path, mode="r", encoding="utf-8")):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [
                    float(value) for value in line[columns["logits"]].split(" ")
                ]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
                )
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
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

            src = src.view(1, src.size(-1))
            seg = seg.view(1, seg.size(-1))

            if args.soft_targets and "logits" in columns.keys():
                self.dataset.append((src, seg, tgt, soft_tgt))
            else:
                self.dataset.append((src, seg, tgt))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def count_labels_num(path):
    """
    It reads the first line of the file and stores the column names in a dictionary, then it reads the
    rest of the file and stores the labels in a set
    
    :param path: the path of the data file
    :return: The number of unique labels in the dataset.
    """
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    """
    If the user has specified a pretrained model, load it. Otherwise, initialize the model with normal
    distribution.
    
    :param args: the arguments that were passed to the script
    :param model: the model we're training
    """
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(
            torch.load(args.pretrained_model_path), strict=False,
        )
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
        )
    else:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
        )
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps * args.warmup
        )
    else:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps * args.warmup, args.train_steps
        )
    return optimizer, scheduler


class Lite(LightningLite):
    def run(self, args):
        seed_everything(args.seed)

        model = Classifier(args)
        load_or_initialize_parameters(args, model)

        train_dataset = ETTrainDataset(args, args.train_path)
        test_dataset = ETTrainDataset(args, args.test_path)

        instances_num = len(train_dataset.dataset)
        args.train_steps = int(instances_num * args.epochs_num / args.batch_size) + 1

        optimizer, scheduler = build_optimizer(args, model)
        model, optimizer = self.setup(model, optimizer)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        )

        train_data, test_data = self.setup_dataloaders(train_loader, test_loader)

        test_acc = Accuracy().to(self.device)
        test_f1 = F1Score(num_classes=args.labels_num, average="micro").to(self.device)

        for epoch in range(args.epochs_num):
            # TRAINING LOOP
            model.train()
            for batch_idx, batch in enumerate(train_data):
                if args.soft_targets:
                    src_batch, seg_batch, tgt_batch, soft_tgt_batch = batch
                    soft_tgt_batch.to(args.device)
                else:
                    src_batch, seg_batch, tgt_batch = batch
                    soft_tgt_batch = None

                optimizer.zero_grad()

                loss, _ = model(src_batch, seg_batch, tgt_batch, soft_tgt_batch)

                self.backward(loss)

                optimizer.step()
                if (batch_idx == 0) or (
                    (batch_idx + 1) % (args.report_steps // args.gpus) == 0
                ):
                    logger.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(src_batch),
                            len(train_loader.dataset) // args.gpus,
                            100.0 * batch_idx / (len(train_loader) // args.gpus),
                            loss.item(),
                        )
                    )

                scheduler.step()

            # TESTING LOOP
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_data:
                    if args.soft_targets:
                        src_batch, seg_batch, tgt_batch, _ = batch
                    else:
                        src_batch, seg_batch, tgt_batch = batch

                    loss, logits = model(src_batch, seg_batch, tgt_batch)
                    test_loss += loss

                    pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
                    gold = tgt_batch

                    test_acc.update(pred, gold)
                    test_f1.update(pred, gold)

            # all_gather is used to aggregated the value across processes
            test_loss = self.all_gather(test_loss).sum() / len(test_loader.dataset)

            logger.info(
                f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({100 * test_acc.compute():.0f}%), F1: ({100 * test_f1.compute():.0f}%)\n"
            )
            test_acc.reset()
            test_f1.reset()

        logger.info("Finished training...")
        if args.output_model_path:
            self.save(model.state_dict(), args.output_model_path)
            logger.info(f"Save trained model: {args.output_model_path}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    finetune_opts(parser)

    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "last"],
        default="first",
        help="Pooling type.",
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
        "--soft_targets", action="store_true", help="Train model with logits."
    )

    parser.add_argument(
        "--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss."
    )

    parser.add_argument("--gpus", type=int, default=2, help="Number of gpus.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.multiprocessing.set_start_method("spawn")

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Training phase.
    logger.info("Start training.")

    Lite(strategy="ddp", accelerator="gpu", devices=args.gpus).run(args)


def get_logger(level=logging.INFO):
    LOG_FORMAT = "[%(asctime)-10s] (line: %(lineno)d) %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger


if __name__ == "__main__":
    logger = get_logger()
    main()
