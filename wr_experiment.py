"""
Word Retrieval Experiment Script

Supported base language:
- eng_Latn (English)

Supported target languages:
- gle_Latn (Irish)
- eus_Latn (Basque)
- cmn_Hans (Chinese)

Experiment configurations:

Data (data):
- Full (full): Use the full dataset
- Single token (single): Use only word pairs where both words are single tokens
- Sample size run for n times (sample):
    - sample_size: How many word pairs to sample per run

Prompt (prompt):
- Zero shot (zero): Standard zero-shot prompting
- Few shot (few): Few-shot prompting with examples

Hidden representation (hidden):
- hidden_base: How to compute hidden states for base language (mean/last)
- hidden_target: How to compute hidden states for target language (mean/last)

Margin Variant (margin_variant):
- ratio: Use ratio-based margin scoring for word retrieval

Output:
Plot denoting the accuracy on word retrieval for each layer

For data sample configuration the plot will contain multiple graphs
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wr_data
import wr_hidden
import wr_plot

LANG_TABLE = {
    "eus_Latn": "Euskara",
    "eng_Latn": "English",
    "gle_Latn": "Gaeilge",
    "cmn_Hans": "中文",
}

# TODO: Builder for this
FEW_SHOT_TEMPLATE = {
    "eus_Latn": """Euskara: "txakurra" - English: "dog"
Euskara: "katua" - English: "cat"
Euskara: "etxe" - English: "house"
Euskara: "mendi" - English: "mountain\"""",
    "gle_Latn": """Gaeilge: "madra" - English: "dog"
Gaeilge: "cat" - English: "cat"
Gaeilge: "teach" - English: "house"
Gaeilge: "sléibhe" - English: "mountain\"""",
    "cmn_Hans": """中文: "狗" - English: "dog"
中文: "猫" - English: "cat"
中文: "房子" - English: "house"
中文: "山" - English: "mountain\"""",
}

def load_model(model_path: str, device_map="auto") -> AutoModelForCausalLM:
    """Load a model"""
    print(f"Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

def run_experiment(
    model,
    tokenizer,
    target_lang: str,
    data: str,
    prompt: str,
    hidden_base: str,
    hidden_target: str,
    margin_variant: str = "ratio",
    data_sample_size: int = 100,
    save_folder: str = "artifacts",
):
    """
    Run a word retrieval experiment with the given configuration.

    Args:
        model: Loaded Hugging Face model
        tokenizer: Corresponding tokenizer
        target_lang (str): Target language ISO code (gle_Latn, eus_Latn, cmn_Hans)
        data (str): Which data to use: "full", "single", "sample"
        prompt (str): Which prompting method: "zero", "few"
        hidden_base (str): "mean" or "last" for base language
        hidden_target (str): "mean" or "last" for target language
        margin_variant (str): Margin variant scoring, default "ratio"
        data_sample_size (int): Number of word pairs to sample per run (for sample data)
        save_folder (str): Folder to save artifacts (JSON, plots)

    Returns:
        None. Output is plot.
    """
    # Load data
    base, target = wr_data.load_sla_word(target_lang)

    if data == "single":
        base, target = wr_data.filter_pairs(base, target, tokenizer)
    elif data == "sample":
        base, target = wr_data.sample_pairs(base, target, sample_size=data_sample_size)

    # Choose hidden state function
    hidden_base_fn = wr_hidden.mean_hidden_state if hidden_base == "mean" else wr_hidden.last_token_hidden_state
    hidden_target_fn = wr_hidden.mean_hidden_state if hidden_target == "mean" else wr_hidden.last_token_hidden_state

    # Prepare few-shot prompt if needed
    if prompt == "few":
        target = [
            f'{FEW_SHOT_TEMPLATE[target_lang]}{LANG_TABLE[target_lang]}: "{t}" - English: "'
            for t in target
        ]

    # Compute layer representations
    base_rep = wr_hidden.layer_representation(
        model=model,
        tokenizer=tokenizer,
        sentences=base,
        device="cuda",
        hidden_state_fn=hidden_base_fn,
    )

    target_rep = wr_hidden.layer_representation(
        model=model,
        tokenizer=tokenizer,
        sentences=target,
        device="cuda",
        hidden_state_fn=hidden_target_fn,
    )

    # Compute accuracy per layer
    layer_acc = {}
    for i in base_rep.keys():
        score = wr_hidden.margin_based_scoring(
            first=base_rep[i], second=target_rep[i], variant=margin_variant
        )
        layer_acc[i] = wr_hidden.layer_accuracy(score)

    # Prepare note text for the plot
    note_text = f"""
Model: {MODEL}
Target Language: {target_lang}
Data: {data}
Prompt: {prompt}
Hidden Base: {hidden_base}
Hidden Target: {hidden_target}
Margin Variant: {margin_variant}
Data Sample Size: {data_sample_size if data=='sample' else 'N/A'}

Actual Data Size: {len(base)}
"""

    # Plot with note
    wr_plot.plot_layer_accuracy(layer_accuracy=layer_acc, save_folder=save_folder, note=note_text)


def main():
    parser = argparse.ArgumentParser(description="Word retrieval task with plotting")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target-lang", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-folder", type=str, default="artifacts", help="Save folder for artifacts for this run")
    parser.add_argument("--data", type=str, default="full", help="Data config for experiment")
    parser.add_argument("--prompt", type=str, default="zero", help="Prompt config for experiment")
    parser.add_argument("--hidden-base", type=str, default="mean", help="Mean over all tokens or last token for base language")
    parser.add_argument("--hidden-target", type=str, default="mean", help="Mean over all tokens or last token for target language")
    parser.add_argument("--data-sample-size", type=int, default=100, help="Number of data to sample (only used when data=sample)")
    parser.add_argument("--margin-variant", type=str, default="ratio")

    args = parser.parse_args()

    global MODEL  # needed for note_text
    MODEL = args.model

    model = load_model(args.model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    run_experiment(
        model=model,
        tokenizer=tokenizer,
        target_lang=args.target_lang,
        data=args.data,
        prompt=args.prompt,
        hidden_base=args.hidden_base,
        hidden_target=args.hidden_target,
        margin_variant=args.margin_variant,
        data_sample_size=args.data_sample_size,
        save_folder=args.save_folder,
    )


if __name__ == "__main__":
    main()
