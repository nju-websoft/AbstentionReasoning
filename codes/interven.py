from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _split_reasoning(output):
    if "</think>" not in output:
        return output.strip(), ""
    reasoning, answer = output.split("</think>", 1)
    return reasoning.strip(), answer.strip()


def _get_probe_target_layer(model, layer_id):
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise AttributeError("Cannot find transformer layers on the loaded model.")
    return layers[layer_id].self_attn.o_proj


def get_input_data(tokenizer, prompt):
    think_token_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
    think_token_id = think_token_ids[0]

    input_text = prompt["input_text"]
    output = prompt["output"]
    reasoning, _ = _split_reasoning(output)

    messages = [{"role": "user", "content": input_text}]
    pre_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if "<think>" not in reasoning and "<think>" not in pre_prompt:
        reasoning = "<think>" + reasoning

    inputs = tokenizer(
        pre_prompt + reasoning,
        return_tensors="pt",
        truncation=True,
        max_length=20000,
    )
    input_ids = inputs["input_ids"]
    think_positions = (input_ids[0] == think_token_id).nonzero(as_tuple=True)[0]
    if len(think_positions) == 0:
        raise ValueError("<think> token not found in tokenized input")

    think_pos = think_positions.item()
    return input_ids[0][think_pos:]


def process_data(prompt, tokenizer):
    think_token_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
    think_token_id = think_token_ids[0]

    input_text = prompt["input_text"]
    output = prompt["output"]
    reasoning, _ = _split_reasoning(output)

    messages = [{"role": "user", "content": input_text}]
    pre_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if "<think>" not in reasoning and "<think>" not in pre_prompt:
        reasoning = "<think>" + reasoning

    inputs = tokenizer(
        pre_prompt + reasoning,
        return_tensors="pt",
        truncation=True,
        max_length=20000,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    think_positions = (input_ids[0] == think_token_id).nonzero(as_tuple=True)[0]
    if len(think_positions) == 0:
        raise ValueError("<think> token not found in tokenized input")

    think_pos = think_positions.item()
    end_pos = attention_mask[0].nonzero(as_tuple=True)[0].max().item()
    return think_pos, end_pos, input_ids, attention_mask


def predict_with_linear_probe(
    datasets,
    model,
    tokenizer,
    layer_id,
    probe,
    model_name=None,
):
    logits_from_probe = None

    def probe_hook(module, input, output):
        nonlocal logits_from_probe
        hidden_states = output.to(dtype=torch.float32)
        with torch.no_grad():
            logits = probe(hidden_states.view(-1, hidden_states.shape[-1]))
            probs = torch.sigmoid(logits).view(hidden_states.shape[:2])
        logits_from_probe = probs.detach().cpu()

    target_layer = _get_probe_target_layer(model, layer_id)
    hook_handle = target_layer.register_forward_hook(probe_hook)
    device = model.device

    final_result = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(datasets):
            logits_from_probe = None
            think_pos, end_pos, input_ids, attention_mask = process_data(data, tokenizer)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if logits_from_probe is None:
                final_result.append([])
                continue

            token_range = list(range(think_pos, end_pos + 1))
            probs = logits_from_probe[:, token_range][0].tolist()
            prefix_probs = [probs[0]] * len(probs)
            for i in range(1, len(prefix_probs)):
                prefix_probs[i] = probs[i] + prefix_probs[i - 1]
            prefix_probs = [prefix_probs[i] / (i + 1) for i in range(len(prefix_probs))]
            final_result.append(prefix_probs)

    hook_handle.remove()
    return final_result


def load_prob_model(model_path, probe_path, layer):
    model_path = str(model_path)
    probe_path = Path(probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe weight not found: {probe_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    target_layer = _get_probe_target_layer(model, layer)
    target_device = next(target_layer.parameters()).device
    probe = LinearProbe(config.hidden_size).to(target_device)
    probe.load_state_dict(torch.load(probe_path, map_location=target_device))
    probe.eval()

    return model, tokenizer, probe
