import torch as th
from dataset import GSMDataset, is_correct
from calculator import sample
from transformers import AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import argparse
import time


def test(model_name, pre_trained=False):
    device = th.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model = None
    if pre_trained:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained("models/" + model_name)
    model.to(device)
    print("Model " + model_name + " loaded.")

    dataset = load_dataset("gsm8k", "main")
    test_samples = dataset["test"][:100]
    questions = test_samples["question"]
    answers = []

    start = time.time()

    for i, qn in enumerate(questions):
        #answer = sample(model, qn, tokenizer, device, sample_len=100)
        #answers.append(answer)
        #if (i + 1) % 5 == 0:
            #print("Iteration " + str(i) + "...")
            # print time in hours, minutes, seconds
            #print("Time elapsed: ", \
            #      time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
        input_ids = tokenizer(qn, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, max_new_tokens=256)
        answers.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        

    # make dataframe of questions and answers, save to csv
    df = pd.DataFrame({"question": questions, "answer": answers})
    df.to_csv("results/" + model_name + ".csv", index=False)

    # calculate accuracy
    correct = 0
    for i in range(len(answers)):
        if is_correct(answers[i], dataset["test"][i]):
            correct += 1

    print("Accuracy: ", correct / len(answers))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--pre_trained", action="store_true")
    args = parser.parse_args()
    model_name = args.model_name
    pre_trained = args.pre_trained

    test(model_name=model_name, pre_trained=pre_trained)
