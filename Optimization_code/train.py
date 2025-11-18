import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AdamW, get_cosine_schedule_with_warmup
from data_loader import IntentDataset, load_jsonl, build_pairs
from models.intent_extractor import IntentExtractor
from losses.sinkhorn_alignment import SinkhornAligner

def collate(batch, tokenizer):
    x_texts = [b["x"] for b in batch]
    y_texts = [b["y"] for b in batch]
    x_inputs = tokenizer(x_texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
    y_inputs = tokenizer(y_texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    return x_inputs, y_inputs, x_texts, y_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--benign_path", type=str, required=True)
    parser.add_argument("--malicious_path", type=str, required=True)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--accum_steps", type=int, default=4)
    args = parser.parse_args()

    lora_cfg = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], "lora_dropout": 0.1}
    extractor = IntentExtractor(args.model_name, lora_cfg)
    tokenizer = extractor.tokenizer
    benign = load_jsonl(args.benign_path, args.text_key)
    malicious = load_jsonl(args.malicious_path, args.text_key)
    canonical = benign + malicious
    pairs = build_pairs(canonical)
    ds = IntentDataset(pairs, tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    aligner = SinkhornAligner()
    nu_emb = aligner.encode(canonical)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.to(device)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    opt = AdamW(extractor.parameters(), lr=args.lr)
    total_steps = args.epochs * math.ceil(len(ds) / args.batch_size)
    warmup_steps = int(0.1 * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    step = 0
    for epoch in range(args.epochs):
        for x_inputs, y_inputs, x_texts, y_texts in dl:
            x_inputs = {k: v.to(device) for k, v in x_inputs.items()}
            y_inputs = {k: v.to(device) for k, v in y_inputs.items()}
            with autocast(enabled=torch.cuda.is_available()):
                out = extractor(x_inputs, y_inputs)
                llm_loss = out.loss
                y_prime = extractor.model.generate(**x_inputs, max_length=x_inputs["input_ids"].shape[1] + 32, do_sample=True)
                y_prime_texts = tokenizer.batch_decode(y_prime, skip_special_tokens=True)
                y_prime_emb = aligner.encode(y_prime_texts)
                loss_align = aligner.loss(y_prime_emb, nu_emb)
                loss = loss_align + args.lambda_align * llm_loss
            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            step += 1

    os.makedirs("checkpoints", exist_ok=True)
    extractor.model.save_pretrained("checkpoints/intent_extractor")
    tokenizer.save_pretrained("checkpoints/intent_extractor")

if __name__ == "__main__":
    main()