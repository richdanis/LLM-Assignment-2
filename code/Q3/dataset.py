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
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.q_max_len = max([len(qn) for qn in self.qns["input_ids"]])
        self.a_max_len = max([len(ans) for ans in self.ans["input_ids"]])

    def __len__(self):
        return len(self.qns["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.qns["input_ids"][idx]
        padding = [0] * (self.q_max_len - len(input_ids))
        attention_mask = [1] * len(input_ids) + padding
        input_ids = input_ids + padding

        # important, padding is not ignored for labels, -100 is the ignore index
        padding = [-100] * (self.a_max_len - len(self.ans["input_ids"][idx]))
        labels = self.ans["input_ids"][idx] + padding

        input_ids = th.tensor(input_ids)
        labels = th.tensor(labels)
        attention_mask = th.tensor(attention_mask)
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)