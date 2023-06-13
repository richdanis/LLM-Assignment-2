class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, ds, loss_on_prefix=True):
        self.qns = ds['question']
        self.ans = ds['answer']
        self.qns = [qn + "\n" for qn in self.qns]
        self.len = len(self.qns)
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix

        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.qns["input_ids"]))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        labels = tokens.copy()
        labels = [-100 if x == 0 else x for x in labels]
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        labels = th.tensor(labels)
        return dict(input_ids=tokens, attention_mask=mask, labels=labels)