import copy
import os
import random
import hashlib
import json
import math
import numpy as np
from extractor import Extractor
from generate import Generate


class MCTSr:
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name
        self.generate = Generate(model_name)

    def sampling_reward(self, query, answer, to_explore_reward):
        if answer not in to_explore_reward:
            to_explore_reward[answer] = []
        reward = self.generate.cal_reward(query, answer)
        to_explore_reward[answer].append(reward)

    def add_to_hints_bank(self, hints, weak_answer, hints_bank):
        if weak_answer not in hints_bank:
            hints_bank[weak_answer] = []
        hints_bank[weak_answer].append(hints)

    def add_to_childs(self, father, child, childs):
        if father not in childs:
            childs[father] = []
        childs[father].append(child)

    def add_to_hints_reward_imp_bank(
        self, hints, weak_answer, reward, answer, hints_reward_imp_bank
    ):
        if weak_answer not in hints_reward_imp_bank:
            hints_reward_imp_bank[weak_answer] = []
        hints_reward_imp_bank[weak_answer].append((hints, reward, answer))

    def get_best_explore_from_ucb(self, to_explore, ucb_bank):
        best_node = None
        highest_ucb = float("-inf")
        for node in to_explore:
            ucb_value = ucb_bank.get(node, float("-inf"))
            if ucb_value > highest_ucb:
                highest_ucb = ucb_value
                best_node = node
        return best_node

    def filter_mature_node(self, childs, to_explore, to_explore_reward, max_expand=3):
        filterd_to_explore = []
        avg_reward = {
            node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2
            for node in to_explore
        }
        for node in to_explore:
            if len(childs.get(node, [])) < max_expand or max(
                [avg_reward.get(child, -999) for child in childs.get(node, [])]
            ) < avg_reward.get(node, -999):
                filterd_to_explore.append(node)
        return filterd_to_explore

    def compute_ucb(self, r_c, N_n, N_c, C):
        return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))

    def update_ucb(
        self, fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4
    ):
        avg_reward = {
            node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2
            for node in to_explore
        }
        leaves = set(to_explore) - set(fathers.values())
        for leaf in leaves:
            ucb_bank[leaf] = self.compute_ucb(
                avg_reward[leaf],
                len(to_explore_reward.get(fathers.get(leaf, None), [])),
                len(to_explore_reward.get(leaf, [])),
                C,
            )
        nodes_to_update = list(leaves)
        while nodes_to_update:
            new_nodes_to_update = set()
            for node in nodes_to_update:
                father = fathers.get(node)
                if father is not None:
                    if father not in ucb_bank:
                        new_nodes_to_update.add(father)
                    if father in ucb_bank:
                        ucb_values = []
                        child_reward = []
                        for child in childs[father]:
                            ucb_values.append(ucb_bank[child])
                            child_reward.append(avg_reward[child])
                        father_reward = (avg_reward[father] + max(child_reward)) / 2
                        ucb_bank[father] = self.compute_ucb(
                            father_reward,
                            len(to_explore_reward.get(fathers.get(father, None), [])),
                            len(to_explore_reward.get(father, [])),
                            C,
                        )
            nodes_to_update = list(new_nodes_to_update)

    def step(self, query, weak_answer, history=[], ans_format=""):
        hints, history = self.generate.get_weak_hints(
            query, weak_answer, history=history, ans_format=ans_format
        )
        answer, history = self.generate.get_better_answer(
            query, weak_answer, hints, history=history, ans_format=ans_format
        )
        return hints, answer, history

    def main_loop(self, query, ground_truth, max_iter=16, ans_format=""):
        to_explore = []
        to_explore_reward = {}
        history_bank = {}
        hints_bank = {}
        ucb_bank = {}
        fathers = {}
        childs = {}
        hints_reward_imp_bank = {}

        ground_truth_label = Extractor.extract_label(ground_truth)
        weak_answer, history = self.generate.get_weak_answer(
            query, ans_format=ans_format
        )
        history_bank[weak_answer] = tuple(history)
        answers_list = [weak_answer]
        to_explore = [weak_answer]
        childs[weak_answer] = []
        fathers[weak_answer] = None
        self.sampling_reward(query, weak_answer, to_explore_reward)

        # Adding a default bad answer for exploration
        total_bad = random.choice(
            [
                "I Don't Know",
                "I can't understand this question.",
                "I can't help with this question.",
                "I don't know how to solve this question.",
                "I don't know the answer to this question.",
                "I don't know the answer to this question, sorry.",
            ]
        )
        total_bad_history = copy.deepcopy(history)
        total_bad_history[-1] = total_bad
        history_bank[total_bad] = tuple(total_bad_history)
        answers_list.append(total_bad)
        to_explore.append(total_bad)
        childs[total_bad] = []
        fathers[total_bad] = None
        self.sampling_reward(query, total_bad, to_explore_reward)

        hints_list = []
        if Extractor.check(ground_truth, weak_answer):
            return (
                hints_list,
                answers_list,
                to_explore,
                to_explore_reward,
                hints_bank,
                history_bank,
                hints_reward_imp_bank,
                fathers,
                childs,
                ucb_bank,
            )

        patient = 0 if "testtime" not in self.data_name else 0
        self.update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank)

        for i in range(max_iter):
            print("iteration:", i)
            filterd_to_explore = self.filter_mature_node(
                childs, to_explore, to_explore_reward
            )
            weak_answer = self.get_best_explore_from_ucb(filterd_to_explore, ucb_bank)
            self.sampling_reward(query, weak_answer, to_explore_reward)
            hints, answer, history = self.step(
                query,
                weak_answer,
                history=history_bank[weak_answer],
                ans_format=ans_format,
            )
            self.add_to_hints_bank(hints, weak_answer, hints_bank)
            history_bank[answer] = tuple(history)
            to_explore.append(answer)
            self.sampling_reward(query, answer, to_explore_reward)
            fathers[answer] = weak_answer
            childs[answer] = []
            self.add_to_childs(weak_answer, answer, childs)
            answers_list.append(answer)
            hints_list.append(hints)

            if Extractor.check(ground_truth, answer):
                return (
                    hints_list,
                    answers_list,
                    to_explore,
                    to_explore_reward,
                    hints_bank,
                    history_bank,
                    hints_reward_imp_bank,
                    fathers,
                    childs,
                    ucb_bank,
                )
            if patient > 0 and Extractor.check(ground_truth, answer):
                patient -= 1

            self.update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank)
            self.add_to_hints_reward_imp_bank(
                hints,
                weak_answer,
                min(to_explore_reward[answer]) - min(to_explore_reward[weak_answer]),
                answer,
                hints_reward_imp_bank,
            )

        return (
            hints_list,
            answers_list,
            to_explore,
            to_explore_reward,
            hints_bank,
            history_bank,
            hints_reward_imp_bank,
            fathers,
            childs,
            ucb_bank,
        )

    def process_dataset(self, dataset):
        for example in dataset:
            self.process_example(example)

    def process_example(self, example):
        file_path = f"{self.data_name}/jsons/{hashlib.md5(str(example).encode()).hexdigest()}.json"
        if os.path.exists(file_path):
            return {}

        query, ground_truth = self.extract_query_and_ground_truth(example)

        if "gsm" in self.data_name:
            ans_format = r'"[Final Answer] The answer is [answer] \n#### [answer]"'
        else:
            ans_format = self.determine_answer_format(ground_truth)

        max_iter = (
            8
            if "meta-math" in self.data_name
            else (2 if "testtime" in self.data_name else 16)
        )

        results = self.main_loop(
            query, ground_truth, max_iter=max_iter, ans_format=ans_format
        )
        if len(results[1]) <= 1 and "rs" in self.data_name:
            return

        data = self.construct_data(query, ground_truth, results, ans_format)

        with open(file_path, "w+") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return data

    def extract_query_and_ground_truth(self, example):
        if "instruction" in example and "output" in example:
            query = example["instruction"] + "\n" + example["input"]
            ground_truth = example["output"]
        elif "context" in example and "question" in example:
            query = (
                (example["context"] + "\n" + example["question"])
                if example["context"]
                else example["question"]
            )
            ground_truth = example["final_answer"][0].replace("$", "")
        elif "GAIC" in self.data_name:
            query = example["problem"]
            ground_truth = example["answer"]
        else:
            query = (
                example.get("query")
                or example.get("problem")
                or example.get("input")
                or example.get("Question")
                or example.get("question")
            )
            ground_truth = (
                example.get("response")
                or example.get("solution")
                or str(example.get("target"))
                or example.get("Answer")
                or example.get("answer")
            )
        return query, ground_truth

    def determine_answer_format(self, ground_truth):
        label = Extractor.extract_label(ground_truth)
        if label.isdigit():
            return r'"[Final Answer] The answer is [number] \n#### [number]"'
        elif label.isalpha() and label.isupper():
            return r'"[Final Answer] The answer is \\boxed{[option]} \n#### [option]"'
        elif label.lower() in ["yes", "no"]:
            return r'"[Final Answer] The answer is \\boxed{[Yes or No]} \n#### [Yes or No]"'
        else:
            return r'"[Final Answer] The answer is \\boxed{[answer formula]} \n#### [answer formula]"'

    def construct_data(self, query, ground_truth, results, hints_prompt):
        (
            hints_list,
            answers_list,
            to_explore,
            to_explore_reward,
            hints_bank,
            history_bank,
            hints_reward_imp_bank,
            fathers,
            childs,
            ucb_bank,
        ) = results
        gt_hints = ""
        data = {
            "query": query,
            "ground_truth": ground_truth,
            "hints_list": hints_list,
            "answers_list": answers_list,
            "ground_truth_hints": gt_hints,
            "hints_prompt": hints_prompt,
            "to_explore": to_explore,
            "to_explore_reward": to_explore_reward,
            "hints_bank": hints_bank,
            "history_bank": history_bank,
            "hints_reward_imp_bank": hints_reward_imp_bank,
            "fathers": fathers,
            "childs": childs,
            "ucb_bank": ucb_bank,
        }
        return data
