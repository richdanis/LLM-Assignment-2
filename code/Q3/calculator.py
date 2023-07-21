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

    decoder_input_ids = [model.config.decoder_start_token_id]
    predicted_ids = []

    for i in range(sample_len):
        with th.no_grad():

            outputs = model(input_ids=toks.input_ids, decoder_input_ids=th.tensor([decoder_input_ids]).to(device))
            logits = outputs.logits[:,i,:]
            # perform argmax on the last dimension (i.e. greedy decoding)
            predicted_id = logits.argmax(-1)
            predicted_ids.append(predicted_id.item())
            # add predicted id to decoder_input_ids
            decoder_input_ids = decoder_input_ids + [predicted_id]
            # stop if EOS token is predicted
            if predicted_id == tokenizer.eos_token_id:
                break

            text = tokenizer.decode(predicted_ids, skip_special_tokens=False)

            if predicted_id.item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            predicted_ids = tokenizer.encode(text, add_special_tokens=True)

    ans = tokenizer.decode(predicted_ids, skip_special_tokens=True)

    return ans
