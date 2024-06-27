import sys
import os
from datasets import load_dataset
from mctsr import MCTSr


def main(model_name, data_name):
    if not os.path.exists(data_name):
        os.mkdir(data_name)
    if not os.path.exists(f"{data_name}/jsons"):
        os.mkdir(f"{data_name}/jsons")

    if "testtime" in data_name:
        if "gsm8k" in data_name:
            if "sample" in data_name:
                dataset = load_dataset("gsm8k", "main", split="test")
                dataset = dataset.select(range(130))
            else:
                dataset = load_dataset("gsm8k", "main", split="test")
        elif "MATH" in data_name:
            dataset = load_dataset("lighteval/MATH", "all", split="test")
    else:
        if "gsmhard" in data_name:
            dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        elif "gsm8k" in data_name:
            if not "mcts" in data_name:
                dataset = load_dataset("gsm8k", "main", split="train")
            else:
                dataset = load_dataset("gsm8k", "main", split="test")
        elif "level5" in data_name:
            dataset = load_dataset(
                "lighteval/MATH", "all", split="test", trust_remote_code=True
            )
            dataset = dataset.filter(lambda example: example["level"].endswith("5"))
        elif "MATH" in data_name and not "level5" in data_name:
            dataset = load_dataset(
                "lighteval/MATH", "all", split="test", trust_remote_code=True
            )
        elif "AIME" in data_name:
            dataset = load_dataset("qq8933/AIME_1983_2024", split="train")
        elif "olympiadbench" in data_name:
            dataset = load_dataset("lmms-lab/OlympiadBench", split="test_en")
            dataset = dataset.filter(
                lambda example: len(example["images"]) == 0
                and example["final_answer"] is not None
                and len(example["final_answer"]) == 1
            )
        elif "meta-math" in data_name:
            dataset = load_dataset("meta-math/MetaMathQA-40K", split="train")
        elif "GAIC" in data_name:
            dataset = load_dataset("qq8933/AGI_Odyssey_MATH_GAIC_2024")
        elif "mathinstruct" in data_name:
            dataset = load_dataset("TIGER-Lab/MathInstruct", split="train")
        else:
            dataset = load_dataset(
                "json",
                data_files=f"/home/bingxing2/ailab/group/ai4phys/math/data_mistral_var_sft.json",
            )

    dataset.shuffle()
    mcts = MCTSr(model_name, data_name)
    mcts.process_dataset(dataset)


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Meta-Llama-3-8B"
    data_name = sys.argv[2] if len(sys.argv) > 2 else "gsm8k"
    main(model_name, data_name)
