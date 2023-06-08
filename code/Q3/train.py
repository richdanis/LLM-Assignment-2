import torch as th
from dataset import GSMDataset
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def train(model_name, lr=3e-4, num_epochs=10, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    
    dataset = load_dataset("gsm8k", "main")

    # split dataset into train and validation
    split = dataset["train"].train_test_split(test_size=0.1)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]

    train_dset = GSMDataset(tokenizer, dataset["train"])
    val_dset = GSMDataset(tokenizer, dataset["validation"])

    device = th.device("cuda")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

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

        train_loss = 0

        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            train_loss += loss.item()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")

        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, device=device)

        print(f"Epoch {epoch} | Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f}")


    model.save_pretrained("models/" + model_name)


def evaluate(model, val_loader, device):

    model.eval()
    val_loss = 0
    with th.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

if __name__ == "__main__":
    train("t5-small")
