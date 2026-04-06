# Thank you Heidi Lei and Katherine Liang!!

import os
import shutil
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, GPT2LMHeadModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "amt-finetuning"

DEFAULT_RUN_NAME = "amt-finetuning-01"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/maestro/ar")
DEFAULT_CKPT_ROOT = "~/amt-finetuning-checkpoints" # CHANGE!!
GPT2_MODEL_NAME = "stanford-crfm/music-medium-800k"
LEARNING_RATE = 1e-5
SEQLEN = 1024

class AmtDataset(Dataset):
    """
    Dataset for loading tokenized event sequences from text files.
    Each line in the file contains integer tokens for a MIDI track.
    """
    def __init__(self, data_dir, split):
        if split in ["val", "valid"]:
            split = "validation"
        data_path = os.path.join(data_dir, f"{split}.txt")
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = [int(x) for x in line.strip().split()]
                # First token is BOS
                if len(tokens) > 1:
                    event_tokens = len(tokens) - 1
                    if event_tokens % 3 != 0:
                        raise ValueError(
                            f"Invalid token count: {len(tokens)} tokens."
                        )
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns input-token and label-token (input = label for LM training)
        return {"input_ids": self.data[idx], "labels": self.data[idx]}


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Fine-tune models on Maestro.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Run name for logging.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing tokenized splits.",
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default=DEFAULT_CKPT_ROOT,
        help="Root directory for checkpoints.",
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="Enable velocity tokens.",
    )
    parser.add_argument(
        "--tokenize-controls",
        action="store_true",
        help="Use expressive vocab with pitchwheel and control-change tokens.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoint directory.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    run_name = args.run_name
    if args.include_velocity and "vel" not in run_name:
        run_name = f"{run_name}-vel"
    if args.tokenize_controls and "exp" not in run_name:
        run_name = f"{run_name}-exp"

    ckpt_dir = os.path.join(args.ckpt_root, run_name)
    os.makedirs(args.ckpt_root, exist_ok=True)

    if os.path.exists(ckpt_dir):
        if args.force:
            shutil.rmtree(ckpt_dir)
        else:
            raise ValueError(f"Checkpoint directory {ckpt_dir} already exists")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL_NAME)

    print("total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # exit()

    ds_train = AmtDataset(
        args.data_dir,
        split="train",
    )

    ds_valid = AmtDataset(
        args.data_dir,
        split="valid",
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=2,
        # per_device_train_batch_size=16,
        per_device_eval_batch_size=2,
        # per_device_eval_batch_size=16,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        max_steps=5000,
        save_steps=500,
        logging_dir="./logs",
        eval_steps=200,
        logging_steps=10,
        bf16=True,  # Enable mixed precision
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=2,
        # dataloader_num_workers=4,
        do_eval=True,
        eval_strategy="steps",
        gradient_accumulation_steps=8,
        # gradient_accumulation_steps=1,
        gradient_checkpointing=True, # to save memory
        save_safetensors=False,
        # eval_on_start=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        optimizers=(optimizer, None),
    )

    # Train
    trainer.train()
