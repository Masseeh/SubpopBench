import torch

def get_bert_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for n, p in network.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    
    total_params = 109483778 # FOR BERT-BASE-UNCASED
    learable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"trainable parameters ratio: {(learable_params / total_params):.4f} ({learable_params} / {total_params})")

    return optimizer


def get_sgd_optim(network, lr, weight_decay):
    optimzer = torch.optim.SGD(
        [p for p in network.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9)
    
    total_params = 109483778 # FOR BERT-BASE-UNCASED
    learable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"trainable parameters ratio: {(learable_params / total_params):.4f} ({learable_params} / {total_params})")

    return optimzer


get_optimizers = {
    "sgd": get_sgd_optim,
    "adamw": get_bert_optim
}
