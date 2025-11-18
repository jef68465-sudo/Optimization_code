import argparse
import torch
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from data_loader import load_jsonl
from models.intent_extractor import IntentExtractor
from models.classifier import SafetyClassifier, train_classifier

def visualize(emb_x, emb_y_prime, emb_y, out_path):
    X = torch.cat([emb_x, emb_y_prime, emb_y], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000)
    Z = tsne.fit_transform(X)
    n1 = emb_x.shape[0]
    n2 = emb_y_prime.shape[0]
    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:n1,0], Z[:n1,1], s=5)
    plt.scatter(Z[n1:n1+n2,0], Z[n1:n1+n2,1], s=5)
    plt.scatter(Z[n1+n2:,0], Z[n1+n2:,1], s=5)
    plt.savefig(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--benign_path", type=str, required=True)
    parser.add_argument("--malicious_path", type=str, required=True)
    parser.add_argument("--text_key", type=str, default="text")
    args = parser.parse_args()

    extractor = IntentExtractor(args.model_name, {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"], "lora_dropout": 0.1})
    tokenizer = extractor.tokenizer
    benign = load_jsonl(args.benign_path, args.text_key)
    malicious = load_jsonl(args.malicious_path, args.text_key)
    x_all = benign + malicious
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor.to(device)
    x_inputs = tokenizer(x_all, truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
    y_prime = extractor.model.generate(**x_inputs, max_length=x_inputs["input_ids"].shape[1] + 32, do_sample=False)
    y_prime_texts = tokenizer.batch_decode(y_prime, skip_special_tokens=True)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    emb_x = encoder.encode(x_all, convert_to_tensor=True, normalize_embeddings=True)
    emb_y_prime = encoder.encode(y_prime_texts, convert_to_tensor=True, normalize_embeddings=True)
    emb_y = encoder.encode(x_all, convert_to_tensor=True, normalize_embeddings=True)
    visualize(emb_x, emb_y_prime, emb_y, "tsne.png")

    y_true = torch.tensor([0] * len(benign) + [1] * len(malicious))
    clf = SafetyClassifier(input_dim=emb_y_prime.shape[1])
    clf = train_classifier(clf, emb_y_prime, y_true, epochs=3, lr=1e-3)
    logits = clf(emb_y_prime)
    y_pred = torch.argmax(logits, dim=1).cpu().tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    asr = sum(y_pred) / len(y_pred) if len(y_pred) > 0 else 0.0
    print({"TPR": tpr, "FPR": fpr, "ASR": asr})

if __name__ == "__main__":
    main()