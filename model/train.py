import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from transformers import AutoTokenizer
from glmf.data.loader import collate_fn,GLMFDataset
from glmf.model.model import XCodeConfig,XCodeModelForCausalLM


def train(args,device):
    ###Load Model
    tokenizer = AutoTokenizer.from_pretrained(f"../CodeQwen1.5-7B-Chat",device_map="auto")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|graph_start|>","<|graph_pad|>", 
                                                            "<|graph_end|>","<|fuzz|>","<|/fuzz|>"]})
    
    config = XCodeConfig(vlmodel=args.model_path, use_lora=args.use_lora, dtype=args.dtype, device_map=device)
    model = XCodeModelForCausalLM(config=config)
    print("###Done Loading Model and Tokenizer")

    ###Load Dataset
    data_list = load_dataset("json",data_files=args.data)
    # graph_list = torch.load(args.graph_data)
    # masks = torch.load(args.graph_masks)
    temp = GLMFDataset(data_list)
    dataloader = DataLoader(temp, batch_size=1, shuffle=True, collate_fn=collate_fn)


    #Training
    # Ensure model is on the correct device and in training mode.
    model.to(device)
    model.gnn.to("cpu")
    
    model.train()
    
    accumulation_steps=4 
    num_epochs=3
    lr=5e-5



    # Create the optimizer after applying the LoRA wrapper.
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    global_step = 0
    loss_track = []
    
    # Zero gradients initially.
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(dataloader):
            print("Batch step:", step)
            batch_loss = 0.0
            batch_size = batch['input']["input_ids"].size(0)
            
            # Process each sample in the batch as a micro-batch.
            for i in range(batch_size):
                global_step += 1

                batch_input = batch['input'].copy()
                if 'token_type_ids' in batch_input:
                    batch_input.pop('token_type_ids')
                    
                micro_input = {
                    "input_ids": batch_input['input_ids'][i].to(device),
                    "attention_mask": batch_input['attention_mask'][i].to(device),
                    "labels": batch_input['labels'][i].to(device),
                }
                
                # Process the graph inputs. If they are tensors, move them to device.
                graph = batch["graph"][i]
                graph_mask = batch["graph_mask"][i]
                
                ##Change Device of graph
                # for key in graph.keys():
                #     graph[key]['feat'] = graph[key]['feat'].to(torch.bfloat16)

                #Change dtype
                # for g in graph.values():
                #     g.ndata['feat'] = g.ndata['feat'].to(torch.bfloat16)

                # Forward pass.
                
                outputs = model(
                    **micro_input,
                    graph=graph,
                    graph_mask=graph_mask,
                )
                # run_nvidia_smi()
                
                # del micro_input
                # del batch_input
                # gc.collect()
                # torch.cuda.empty_cache()
                
                # run_nvidia_smi()
                
                loss = outputs.loss
                loss = loss / accumulation_steps
                loss.backward()
                batch_loss += outputs.loss.item()  # For logging (using the unscaled loss).

                # Update parameters once enough gradients have been accumulated.
                if global_step % accumulation_steps == 0:
                    optimizer.step()       # Update parameters.
                    optimizer.zero_grad()  # Reset gradients.
                    
            # Log average loss for this batch.
            avg_batch_loss = batch_loss / batch_size
            loss_track.append(avg_batch_loss)
            print(f"Batch {step}: loss = {avg_batch_loss:.4f}")

    if model.config.use_lora == True:
        model.model = model.model.merge_and_unload()
        
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Train the XCode model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model directory (for VL model).")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the JSON dataset file.")
    parser.add_argument("--graph_data", type=str, required=True,
                        help="Path to the graph data file (torch format).")
    parser.add_argument("--graph_masks", type=str, required=True,
                        help="Path to the graph masks file (torch format).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model and tokenizer.")
    parser.add_argument("--use_lora", action="store_true",default = False,
                        help="Flag to enable LoRA training modifications.")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Data type to use (e.g., float32 or bfloat16).")
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    #                     help="Device to train on (e.g., 'cuda' or 'cpu').")
    
    args = parser.parse_args()
    
    train(args, device)
    


    
            




