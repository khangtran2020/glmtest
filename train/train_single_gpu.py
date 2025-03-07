import types
import torch
from argparse import Namespace
from transformers import TrainingArguments, Trainer, PreTrainedTokenizer
from data.loader import GLMFDataset, collate_fn
from model.model import GLMFModelForCausalLM
from typing import Dict
from train.utils import get_train_dataloader


def initialize_trainer_single_gpu(
    args: Namespace,
    train_dataset: GLMFDataset,
    valid_dataset: GLMFDataset,
    model: GLMFModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
) -> Trainer:

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        save_strategy="steps",
        save_steps=500,
        optim="adamw_torch",
        logging_dir=args.output_dir,
        logging_steps=100,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=None,
    )

    trainer.get_train_dataloader = types.MethodType(get_train_dataloader, trainer)

    return trainer


# import torch
# from transformers import AdamW
# from transformers import PreTrainedTokenizer
# from model.model import GLMFModelConfig, GLMFModelForCausalLM


# def train(
#     data_loader: torch.utils.data.DataLoader,
#     dtype: str,
#     use_lora: bool,
#     llm_path: str,
#     save_path: str,
#     tokenizer: PreTrainedTokenizer,
#     device: torch.device,
# ):
#     ###Load Model
#     config = GLMFModelConfig(
#         llm_model=llm_path,
#         use_lora=use_lora,
#         dtype=dtype,
#         device_map=device,
#     )
#     model = GLMFModelForCausalLM(config=config)
#     print("###Done Loading Model and Tokenizer")

#     # Training
#     # Ensure model is on the correct device and in training mode.
#     model.to(device)
#     model.gnn.to("cpu")
#     model.train()

#     accumulation_steps = 4
#     num_epochs = 3
#     lr = 5e-5

#     # Create the optimizer after applying the LoRA wrapper.
#     optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#     global_step = 0
#     loss_track = []

#     # Zero gradients initially.
#     optimizer.zero_grad()

#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         for step, batch in enumerate(data_loader):
#             print("Batch step:", step)
#             batch_loss = 0.0
#             batch_size = batch["input"]["input_ids"].size(0)

#             # Process each sample in the batch as a micro-batch.
#             for i in range(batch_size):
#                 global_step += 1

#                 batch_input = batch["input"].copy()
#                 if "token_type_ids" in batch_input:
#                     batch_input.pop("token_type_ids")

#                 micro_input = {
#                     "input_ids": batch_input["input_ids"][i].to(device),
#                     "attention_mask": batch_input["attention_mask"][i].to(device),
#                     "labels": batch_input["labels"][i].to(device),
#                 }

#                 graph = batch["graph"][i]
#                 graph_mask = batch["graph_mask"][i]

#                 outputs = model(
#                     **micro_input,
#                     graph=graph,
#                     graph_mask=graph_mask,
#                 )

#                 loss = outputs.loss
#                 loss = loss / accumulation_steps
#                 loss.backward()
#                 batch_loss += outputs.loss.item()

#                 if global_step % accumulation_steps == 0:
#                     optimizer.step()
#                     optimizer.zero_grad()

#             # Log average loss for this batch.
#             avg_batch_loss = batch_loss / batch_size
#             loss_track.append(avg_batch_loss)
#             print(f"Batch {step}: loss = {avg_batch_loss:.4f}")

#     if model.config.use_lora == True:
#         model.model = model.model.merge_and_unload()

#     model.save_pretrained(save_path)
#     tokenizer.save_pretrained(save_path)
