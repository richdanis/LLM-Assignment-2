import re
import torch as th

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, ds):
        self.qns = ds['question']
        self.ans = ds['answer']
        self.qns = [qn for qn in self.qns]
        self.ans = [ans for ans in self.ans]
        self.qns = tokenizer(self.qns, padding='longest', return_tensors="pt")
        self.ans = tokenizer(self.ans, padding=False)
        self.a_max_len = max([len(an) for an in self.ans["input_ids"]])

    def __len__(self):
        return len(self.qns["input_ids"])

    def __getitem__(self, idx):

        # important, padding is not ignored for labels, -100 is the ignore index
        padding = [-100] * (self.a_max_len - len(self.ans["input_ids"][idx]))
        labels = self.ans["input_ids"][idx] + padding

        labels = th.tensor(labels)

        return dict(input_ids=self.qns["input_ids"][idx], \
                     attention_mask=self.qns["attention_mask"][idx], \
                        labels=labels)