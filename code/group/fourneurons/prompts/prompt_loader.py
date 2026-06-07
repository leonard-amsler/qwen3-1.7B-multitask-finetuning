"""Utility for loading prompts from text files."""

def load_prompt(prompt_file, verbose=False):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

        if verbose:
            print(f"\nLoaded prompt from {prompt_file} (length {len(prompt)} characters)")
            print("\nLOADED PROMPT:\n")
            print(prompt)

        return prompt

SP_GENERAL_QCM_THINK = (
    "You are a helpful assistant. For multiple-choice questions, "
    "reason step by step, then provide your final answer as a single "
    r"letter inside \boxed{}, for example: \boxed{A}. "
    r"Do not include anything after the \boxed{}."
)

if __name__ == "__main__":
    prompt = load_prompt("/scratch/nico/standard-project-m2-4neurons/prompts/sp_general_qcm_think.txt", verbose=True)

    assert prompt == SP_GENERAL_QCM_THINK, "Loaded prompt from 'sp_general_qcm_think.txt' does not match expected SP_GENERAL_QCM_THINK"
    print("\nPrompt loaded successfully and matches expected content.")

