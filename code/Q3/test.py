import torch as th
from dataset import get_examples, GSMDataset
from calculator import sample
from transformers import AutoTokenizer, T5ForConditionalGeneration


def main():
    device = th.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained("model_ckpts")
    model.to(device)
    print("Model Loaded")

    test_examples = get_examples("test")
    qn = test_examples[1]["question"]
    sample_len = 100
    print(qn.strip())
    print(sample(model, qn, tokenizer, device, sample_len))


if __name__ == "__main__":
    main()
