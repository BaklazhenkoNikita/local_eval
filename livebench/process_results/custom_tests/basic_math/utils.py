import re
import warnings
from livebench.process_results.util import last_boxed_only_string, remove_boxed

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for basic_math grading. "
        "Please install sympy via pip install sympy"
    )


def basic_math_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    """
    Grade basic math problems.

    Args:
        ground_truth: The correct answer
        llm_answer: The model's answer
        debug: If True, print debug information

    Returns:
        1 if correct, 0 if incorrect
    """
    retval = 0
    parsed_answer = None

    if isinstance(ground_truth, list):
        ground_truth = ground_truth[-1]

    llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
    llm_answer = llm_answer.replace("\\dfrac", "\\frac")
    llm_answer = llm_answer.replace("\\tfrac", "\\frac")
    llm_answer = llm_answer.replace("\n", " ")

    last_boxed = last_boxed_only_string(llm_answer)
    if last_boxed:
        parsed_answer = normalize_final_answer(remove_boxed(last_boxed))

    if parsed_answer is None:
        last_line = llm_answer.split('\n')[-1]
        if last_line.count('$') >= 2:
            close_pos = last_line.rfind('$')
            if close_pos > 0 and last_line[close_pos - 1] == '$':
                close_pos -= 1
            open_pos = last_line.rfind('$', 0, close_pos)
            math = last_line[open_pos + 1:close_pos]
            if '=' in math:
                math = math.split('=')[-1].strip()
            parsed_answer = normalize_final_answer(math)

    if parsed_answer is not None:
        try:
            if is_equiv(ground_truth, parsed_answer):
                retval = 1
        except Exception as e:
            warnings.warn(f"Error when comparing ground truth and parsed answer: {e}")

    if retval == 0 and parsed_answer is None:
        if len(llm_answer) > 0 and llm_answer[-1] == '.':
            llm_answer = llm_answer[:-1]
        if ground_truth == llm_answer[-len(ground_truth):]:
            retval = 1

    if debug and retval == 0:
        print('INCORRECT')
        print('GROUND TRUTH:', ground_truth)
        if parsed_answer:
            print('PARSED ANSWER:', parsed_answer)
        print('RAW ANSWER:', llm_answer)

    return retval


def is_equiv(x1: str, x2: str) -> bool:
    """
    Check if two math expressions are equivalent.

    Args:
        x1: First expression (ground truth)
        x2: Second expression (parsed answer)

    Returns:
        True if equivalent, False otherwise
    """
    try:
        if x1.strip() == x2.strip():
            return True

        try:
            parsed_x1 = parse_latex(x1)
            parsed_x2 = parse_latex(x2)
        except:
            try:
                parsed_x1 = sympy.sympify(x1)
                parsed_x2 = sympy.sympify(x2)
            except:
                return x1.replace(" ", "") == x2.replace(" ", "")

        try:
            diff = parsed_x1 - parsed_x2
            if sympy.simplify(diff) == 0:
                return True
        except:
            pass

        try:
            if sympy.Abs(sympy.simplify(diff)) < 0.001:
                return True
        except:
            pass

        return False
    except Exception as e:
        warnings.warn(f"Failed comparing {x1} and {x2}: {e}")
        return False


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a basic math question.

    Args:
        final_answer: The raw answer string

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1].strip()

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{\[])", "sqrt{\\2}", final_answer)

    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").replace(".", "").replace("-", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()
