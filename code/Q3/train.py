import torch as th
from dataset import GSMDataset
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def train(model_name, lr=1e-5, num_epochs=3, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    
    dataset = load_dataset("gsm8k", "main")
    train_dset = GSMDataset(tokenizer, dataset["train"])

    device = th.device("cuda")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")

    model.save_pretrained("models/" + model_name + "_original")


if __name__ == "__main__":
    train("t5-small")
