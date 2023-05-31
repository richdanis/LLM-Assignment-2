import torch as th
from dataset import extract_answer, GSMDataset, is_correct
from calculator import sample
from transformers import AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def test(model_name, pretrained=False):
    device = th.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model = None
    if pretrained:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained("models/" + model_name)
    model.to(device)
    print("Model Loaded")

    dataset = load_dataset("gsm8k", "main")
    test_samples = dataset["test"][:100]
    questions = test_samples["question"]
    answers = []

    for qn in tqdm(questions):
        answer = sample(model, qn, tokenizer, device, sample_len=100)
        answers.append(answer)

    # make dataframe of questions and answers, save to csv
    df = pd.DataFrame({"question": questions, "answer": answers})
    df.to_csv("results/" + model_name + ".csv", index=False)

    # calculate accuracy
    correct = 0
    for i in range(len(answers)):
        if is_correct(answers[i], test_samples[i]):
            correct += 1
    print("Accuracy: ", correct / len(answers))

if __name__ == "__main__":
    test()
