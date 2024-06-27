class Extractor:
    @staticmethod
    def extract_matching_bracket(target_str: str):
        if not target_str:
            return target_str
        current_nest_level = 1
        for i, ch in enumerate(target_str):
            if ch == "{":
                current_nest_level += 1
            elif ch == "}":
                current_nest_level -= 1
            if current_nest_level == 0:
                break
        return target_str[:i]

    @staticmethod
    def clean(target_str: str):
        opt = target_str.strip().replace("{{", "{").replace("}}", "}")
        if not opt:
            return opt
        if opt[-1] == "." or opt[-1] == "。":
            return opt[:-1]
        return opt

    @staticmethod
    def extract_answer(pred: str, extract_last_num=False):
        if pred.find("The final answer is ") >= 0:
            x = pred[pred.find("The final answer is ") + len("The final answer is ") :]
            x = x[1 : x.find("$.")]
            return Extractor.clean(x)
        if pred.find("\n\nQuestion:") >= 0:
            pred = pred.split("\n\nQuestion:")[0]
            if pred.find("The answer is"):
                pred = pred[pred.find("The answer is") + len("The answer is") :]
                return Extractor.clean(pred)
        if pred.find("# Answer") >= 0:
            return Extractor.clean(pred[pred.find("# Answer") + len("# Answer") :])
        if pred.find("The answer is:") >= 0:
            return Extractor.clean(
                pred[pred.find("The answer is:") + len("The answer is:") :]
            )
        if pred.find("####") >= 0:
            return Extractor.clean(pred[pred.find("####") + 4 :])
        left = "\\boxed{"
        if pred.find(left) >= 0:
            pred = pred[pred.find(left) + len(left) :]
            return Extractor.clean(Extractor.extract_matching_bracket(pred))

        if extract_last_num:
            nums = []
            opt = ""

            def contain_digit(opt):
                for ch in opt:
                    if ch.isdigit():
                        return True
                return False

            for ch in pred:
                if ch.isdigit() or ch in " ,.":
                    opt = opt + ch
                else:
                    if contain_digit(opt):
                        nums.append(opt)
                    opt = ""
            if contain_digit(opt):
                return Extractor.clean(opt)
            if nums:
                return Extractor.clean(nums[-1])
        return None
