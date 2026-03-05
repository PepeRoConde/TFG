import torch


def print_prediccions(output, target):
    probs = torch.nn.functional.softmax(output, dim=1)

    num_pred = 25
    batch_size = probs.shape[0]

    # Random sample without replacement
    idx = torch.randperm(batch_size)[:num_pred]

    preds = probs[idx, 1].detach().cpu().numpy()
    labels = (
        (target[idx, 1] if target.dim() > 1 else target[idx]).detach().cpu().numpy()
    )

    pred_str = ", ".join(f"{p:0.2f}" for p in preds)
    label_str = ",  ".join(f"{int(l):3d}" for l in labels)

    print("Prediccions de mostra (random):")
    print(f"  prediccion : [{pred_str}]")
    print(f"  etiquetas  : [{label_str}]")
