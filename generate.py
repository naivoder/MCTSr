import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generate:
    def __init__(self, model_name, dataset_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.dataset_name = dataset_name

    def generate(self, prompt, history=[], timeout=150, truncate=True):
        if "testtime" in self.dataset_name:
            timeout = 150
        print("awaiting response...")
        history_ = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": h}
            for i, h in enumerate(history)
        ]
        if truncate:
            history_ = history_[-2:]

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids, max_length=512, temperature=0.95, num_return_sequences=1
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"response received!")
        return response, list(history) + [prompt, response]

    def cal_reward(self, query, ans):
        ret = self.generate(query)
        score = ret[0].split("Score")[-1]
        scores = re.findall(r"\-?\d+\.\d+|\-?\d+", score)
        if not scores:
            raise Exception("no")
        else:
            ret = float(scores[-1])
            if ret >= 95:
                ret = 50
            return ret

    def get_weak_answer(self, query, new_len=0, ans_format=""):
        query = f"Question: {query}\nThe response should begin with [reasoning process]...[Verification]... and end with {ans_format}\nLet's think step by step."
        return self.generate(query, timeout=90)

    def get_weak_hints(
        self,
        query,
        weak_answer,
        ground_truth_label=None,
        new_len=0,
        history=[],
        alreadygood=False,
        ans_format="",
    ):
        query = f"Question: {query}\nSince we have a weak Answer, could you provide me with a relection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score!\nLet's think step by step."
        return self.generate(query, history)

    def get_better_answer(
        self, query, weak_answer, hint, new_len=0, history=[], ans_format=""
    ):
        query = f"Question: {query}\nPlease refine the your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with end with {ans_format}\nLet's think step by step."
        return self.generate(query, history)

    def get_gt_hints(self, query, ground_truth, new_len=0):
        query = f"Question: {query}\nGround Truth:{ground_truth}\nAccording to ground truth answer we have, Could you descript the thought process of ground truth answer, please don’t give me the answer, just the thought process?"
        return self.generate(query)
