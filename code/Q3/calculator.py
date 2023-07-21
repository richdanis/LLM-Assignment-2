from contextlib import contextmanager
import signal
import torch as th

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    EQUALS_TOKENS = set([28, 796, 47505])

    toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
    ans = None
    out = None

    for _ in range(sample_len):
        with th.no_grad():

            if ans is None:
                out = model.generate(
                    **toks, max_length=1, pad_token_id=model.config.eos_token_id
                )
            else:
                # get decoder input ids from ans
                dec_inputs = tokenizer([ans], padding=False, return_tensors="pt").to(device)["input_ids"]
                # pass dec_inputs as decoder input
                out = model.generate(
                    **toks,
                    max_length=1,
                    pad_token_id=model.config.eos_token_id,
                    decoder_input_ids=dec_inputs,
                )
            text = tokenizer.batch_decode(out)[0]

            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            ans = text
    return qn
