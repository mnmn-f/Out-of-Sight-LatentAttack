from attacks import (
    apply_question_attack,
    apply_role_attack,
    build_mi_adversary_user_prompt,
    is_mi_role_attack,
    mi_system_prompt,
)


def _default_system_message(args=None) -> str:
    model_name = (getattr(args, "model_name", "") or "").lower()
    if "qwen" in model_name:
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    return "You are a helpful assistant."


def _attacker_override(role: str, question: str, gold: str, args):
    if not is_mi_role_attack(role, args):
        return None
    return [
        {"role": "system", "content": mi_system_prompt(args)},
        {"role": "user", "content": build_mi_adversary_user_prompt(question, gold, args)},
    ]


def _math_or_science_final_instruction(args) -> str:
    if args.task in ["arc_easy", "arc_challenge", "openbookqa", "gpqa", "medqa"]:
        return (
            "Your final answer must be selected from A,B,C,D. "
            "For example \\boxed{{A}}. Do not add any other contents inside the box."
        )
    if args.task in ["winogrande"]:
        return (
            "Your final answer must be selected from 1 and 2. "
            "For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."
        )
    return "Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."


def _code_final_instruction() -> str:
    return (
        "You must put all python code as a self-contained Python function in markdown code blocks. "
        "For example ```python\nimport math\ndef add(a, b):\n    return a + b```."
    )


def _sequential_user_template(role: str, question_token: str, args) -> str:
    if role == "planner":
        return f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question_token}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""

    if role == "critic":
        return f"""
Question: {question_token}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    if role == "refiner":
        return f"""
Question: {question_token}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""

    if role == "judger":
        if args.task in ["mbppplus", "humanevalplus"]:
            final_instruction = _code_final_instruction()
            answer_format = "Now, reason step by step and output the final answer inside ```python\nYOUR_PYTHON_CODE\n```."
        else:
            final_instruction = _math_or_science_final_instruction(args)
            answer_format = "Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."
        return f"""
Target Question: {question_token}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
{final_instruction}

{answer_format}
"""

    raise ValueError(f"Unsupported role: {role}")


def build_agent_message_sequential_latent_mas(role: str, question: str, context: str = "", method=None, args=None, gold: str = ""):
    question_token = "<<QUESTION_PLACEHOLDER>>"
    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    question = apply_question_attack(question, args)

    override = _attacker_override(role, question, gold, args)
    if override is not None:
        return override

    user_template = _sequential_user_template(role, question_token, args)
    user_prompt = user_template.replace(question_token, question)
    user_prompt = apply_role_attack(user_prompt, role, args)
    return [
        {"role": "system", "content": _default_system_message(args)},
        {"role": "user", "content": user_prompt},
    ]


def _hierarchical_role_instruction(role: str, args) -> str:
    if args.task in ["mbppplus", "humanevalplus"]:
        role_map = {
            "planner": "You are a math agent. Given the programming problem, provide your solution as a self-contained Python function.",
            "critic": "You are a science agent. Given the programming problem, provide your solution as a self-contained Python function.",
            "refiner": "You are a code agent. Given the programming problem, provide your solution as a self-contained Python function.",
            "judger": "You are a task summarizer. Given the programming problem and responses from previous agents as reference, provide the final Python solution.",
        }
        return role_map[role]

    final_instruction = _math_or_science_final_instruction(args)
    role_map = {
        "planner": "You are a math agent.",
        "critic": "You are a science agent.",
        "refiner": "You are a code agent.",
        "judger": "You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step.",
    }
    return f"{role_map[role]} Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.\n{final_instruction}"


def build_agent_message_hierarchical_latent_mas(role: str, question: str, context: str = "", method=None, args=None, gold: str = ""):
    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    question = apply_question_attack(question, args)

    override = _attacker_override(role, question, gold, args)
    if override is not None:
        return override

    instruction = _hierarchical_role_instruction(role, args)
    user_content = f"""
{instruction}

Input Question: {question}

Your response:
"""
    user_content = apply_role_attack(user_content, role, args)
    return [
        {"role": "system", "content": _default_system_message(args)},
        {"role": "user", "content": user_content},
    ]
