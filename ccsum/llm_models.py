from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
import tqdm

logger = logging.getLogger(__name__)


def load_model(model_name="google/flan-t5-xxl"):
    logger.info(f"Loading {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    logger.info("Model and tokenizer loaded.")
    return model, tokenizer


def query(
        model, tokenizer, prompts,
        decoding_args={
            "max_new_tokens": 16, "do_sample": True, "top_k": 100, "top_p": 0.8, "temperature": 0.7,
            "num_return_sequences": 1}):
    with torch.no_grad():
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to('cuda')
        generated_outputs = model.generate(input_ids, **decoding_args)
        if decoding_args.get("return_dict_in_generate", False) is True:
            return generated_outputs
        else:
            return tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)


def batch_query(model, tokenizer, prompts, decoding_args, batch_size=8):
    dataloader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False, num_workers=1)
    outputs = []

    for d in tqdm.tqdm(dataloader):
        output = query(model, tokenizer, d, decoding_args)
        if isinstance(output, list):
            outputs.extend(output)
        else:
            outputs.append(output)
    return outputs
