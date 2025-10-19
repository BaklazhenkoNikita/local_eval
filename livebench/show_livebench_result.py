"""
Usage:
python3 show_livebench_result.py
"""
import argparse
import pandas as pd
import glob
import os
import re
import numpy as np

from livebench.common import (
    LIVE_BENCH_RELEASES,
    get_categories_tasks,
    load_questions,
    load_questions_jsonl
)
from livebench.model import get_model_config


def show_individual_test_summary(args, df, questions_all, bench, model_filter, valid_question_ids):
    """Show summary for a single test."""
    summary_data = {}

    # Load model answer files for this specific benchmark
    answer_files = glob.glob(f"data/{bench}/**/model_answer/*.jsonl", recursive=True)

    if not answer_files:
        return  # Silently skip if no results

    for answer_file in answer_files:
        if os.path.exists(answer_file):
            answers = pd.read_json(answer_file, lines=True)

            if len(answers) == 0 or 'model_id' not in answers.columns:
                continue

            # Filter to only include valid question IDs
            answers = answers[answers['question_id'].isin(valid_question_ids)]

            if len(answers) == 0:
                continue

            model_id = answers['model_id'].iloc[0]

            # Skip if we're filtering by model list and this model isn't in it
            if model_filter is not None and model_id.lower() not in model_filter:
                continue

            # Find matching model name from df
            matching_models = [m for m in set(df["model"]) if isinstance(m, str) and m.lower() == model_id.lower()]

            for model in matching_models:
                if model in summary_data:
                    continue

                model_summary = {}

                # Get score for this model
                model_scores = df[df["model"] == model]["score"]
                if len(model_scores) > 0:
                    avg_score = model_scores.mean()
                    model_summary['Score'] = round(avg_score, 1)

                # Number of questions answered
                model_summary['Questions'] = len(answers)

                # Total tokens
                if 'total_output_tokens' in answers.columns:
                    valid_tokens = answers[answers['total_output_tokens'] != -1]['total_output_tokens']
                    if len(valid_tokens) > 0:
                        model_summary['Total Tokens'] = int(valid_tokens.sum())

                # Total time
                total_time = None
                if 'total_inference_time_seconds' in answers.columns:
                    valid_times = answers[pd.notna(answers['total_inference_time_seconds'])]['total_inference_time_seconds']
                    if len(valid_times) > 0:
                        total_time = valid_times.sum()

                if total_time is None and 'tstamp' in answers.columns:
                    timestamps = answers['tstamp'].dropna()
                    if len(timestamps) > 1:
                        total_time = timestamps.max() - timestamps.min()

                if total_time is not None and total_time > 0:
                    model_summary['Total Time (s)'] = round(total_time, 2)

                    if 'Total Tokens' in model_summary:
                        avg_toks = model_summary['Total Tokens'] / total_time
                        model_summary['Avg Tok/s'] = round(avg_toks, 2)

                if model_summary:
                    summary_data[model] = model_summary

    if summary_data:
        summary_df = pd.DataFrame(summary_data).T
        col_order = ['Score', 'Questions', 'Total Tokens', 'Total Time (s)', 'Avg Tok/s']
        available_cols = [col for col in col_order if col in summary_df.columns]
        summary_df = summary_df[available_cols]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(summary_df)


def calculate_usage(args, df, questions_all, release_set):
    """Calculate average token usage and timing metrics for all answers by task and category."""

    # Get the set of valid question IDs
    valid_question_ids = {q['question_id'] for q in questions_all}
    
    # Get model list to filter by if provided
    model_filter = None
    if args.model_list is not None:
        model_filter = {get_model_config(x).display_name.lower() for x in args.model_list}
        print(f"Filtering token usage for models: {', '.join(sorted(model_filter))}")
    
    # Load model answer files
    model_answers = {}
    for bench in args.bench_name:
        # Find all model answer files without filtering by model name
        answer_files = glob.glob(f"data/{bench}/**/model_answer/*.jsonl", recursive=True)
        
        for answer_file in answer_files:
            # Load the answer file
            if os.path.exists(answer_file):
                answers = pd.read_json(answer_file, lines=True)
                
                # Skip if empty or doesn't have model_id column
                if len(answers) == 0 or 'model_id' not in answers.columns:
                    continue
                
                # Filter to only include valid question IDs
                answers = answers[answers['question_id'].isin(valid_question_ids)]
                
                # Skip if empty after filtering
                if len(answers) == 0:
                    continue
                
                # Group answers by model_id
                grouped_answers = answers.groupby('model_id')
                
                # Process each model group
                for model_id, model_group in grouped_answers:
                    # Check if this model_id matches any model in our correct answers
                    if not isinstance(model_id, str):
                        continue
                    
                    # Skip if we're filtering by model list and this model isn't in it
                    if model_filter is not None and model_id.lower() not in model_filter:
                        continue
                    
                    matching_models = [m for m in set(df["model"]) if isinstance(m, str) and m.lower() == model_id.lower()]
                    
                    for model in matching_models:
                        if model not in model_answers:
                            model_answers[model] = model_group
                        else:
                            model_answers[model] = pd.concat([model_answers[model], model_group], ignore_index=True)
    
    # Create dataframe for token usage
    usage_data = []
    
    # Process each model
    for model, answers_df in model_answers.items():
        # Check if total_output_tokens exists in the dataframe
        if 'total_output_tokens' not in answers_df.columns:
            print(f"Model {model} missing total_output_tokens data")
            continue
            
        # Filter answers to only include those with token data and where total_output_tokens is not -1
        valid_answers = answers_df.dropna(subset=['total_output_tokens'])
        valid_answers = valid_answers[valid_answers['total_output_tokens'] != -1]
        
        # Get all answers for this model
        model_all = df[df["model"] == model]
        
        # Process all answers
        for _, judgment in model_all.iterrows():
            question_id = judgment["question_id"]
            
            # Skip if question_id not in valid_question_ids
            if question_id not in valid_question_ids:
                continue
                
            matching_answer = valid_answers[valid_answers["question_id"] == question_id]
            
            if len(matching_answer) == 0:
                continue
                
            # Add to usage data
            answer_row = matching_answer.iloc[0]
            usage_entry = {
                "model": model,
                "question_id": question_id,
                "task": judgment["task"],
                "category": judgment["category"],
                "total_output_tokens": answer_row["total_output_tokens"]
            }

            # Add timing information if available
            if "total_inference_time_seconds" in answer_row and pd.notna(answer_row["total_inference_time_seconds"]):
                usage_entry["inference_time"] = answer_row["total_inference_time_seconds"]

            if "average_tokens_per_second" in answer_row and pd.notna(answer_row["average_tokens_per_second"]):
                usage_entry["tokens_per_second"] = answer_row["average_tokens_per_second"]

            usage_data.append(usage_entry)
    
    if not usage_data:
        return  # No data available, skip silently
        
    # Create dataframe from collected data
    usage_df = pd.DataFrame(usage_data)
    
    # Calculate average by task
    task_usage = usage_df.groupby(["model", "task"])["total_output_tokens"].mean().reset_index()
    task_pivot = pd.pivot_table(task_usage, index=['model'], values="total_output_tokens", columns=["task"])
    
    # Calculate average by category
    category_usage = usage_df.groupby(["model", "task", "category"])["total_output_tokens"].mean().reset_index()
    
    # Get tasks per category from the raw df
    tasks_by_category = {}
    for _, row in df.iterrows():
        category = row["category"]
        task = row["task"]
        if category not in tasks_by_category:
            tasks_by_category[category] = set()
        tasks_by_category[category].add(task)

    # Calculate averages
    category_pivot = pd.pivot_table(
        category_usage, 
        index=['model'], 
        values="total_output_tokens", 
        columns=["category"]
    )
    
    # Check each model and category - if not all tasks in category have data, mark as NaN
    for model in category_pivot.index:
        for category, tasks in tasks_by_category.items():
            if category in category_pivot.columns:
                # Get tasks for this model and category in all answers
                tasks_with_data = set(usage_df[(usage_df["model"] == model) & 
                                                    (usage_df["category"] == category)]["task"])
                # If any task is missing, set to NaN
                if not tasks.issubset(tasks_with_data):
                    category_pivot.at[model, category] = np.nan
    
    # Calculate averages
    avg_by_model = {}
    
    # List of all categories
    all_categories = list(category_pivot.columns)
    
    for model in category_pivot.index:
        # Check if any category has a NaN value
        has_missing_category = any(pd.isna(category_pivot.at[model, cat]) for cat in all_categories if cat in category_pivot.columns)
        
        # Only calculate average if all categories have data
        if not has_missing_category:
            values = [category_pivot.at[model, cat] for cat in all_categories if cat in category_pivot.columns]
            avg_by_model[model] = sum(values) / len(values)
    
    # Add average column
    category_pivot['average'] = pd.Series({
        model: avg_by_model.get(model, 0) 
        for model in category_pivot.index
        if model in avg_by_model
    })
    
    # Sort by average
    # Models with complete data come first (sorted by their averages)
    # Then models with incomplete data (sorted alphabetically)
    models_with_average = [model for model in category_pivot.index if model in avg_by_model]
    models_without_average = [model for model in category_pivot.index if model not in avg_by_model]
    
    sorted_models_with_average = sorted(
        models_with_average,
        key=lambda model: avg_by_model.get(model, 0),
        reverse=True
    )
    
    sorted_models = sorted_models_with_average + sorted(models_without_average)
    category_pivot = category_pivot.reindex(sorted_models)
    
    # Move average to first column
    first_col = category_pivot.pop('average')
    category_pivot.insert(0, 'average', first_col)
    
    # Format values as decimals for better readability
    category_pivot = category_pivot.round(1)
    task_pivot = task_pivot.round(1)
    
    # Save to CSV
    task_pivot.to_csv('task_usage.csv')
    category_pivot.to_csv('group_usage.csv')
    
    # Calculate and display benchmark summary statistics
    # If multiple benchmarks, show individual results first
    if len(args.bench_name) > 1:
        print("\n########## Individual Test Results ##########")
        for bench in args.bench_name:
            # Load questions for this specific benchmark
            bench_questions = []
            list_of_question_files = []
            original_question_file = f"data/{bench}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(f"data/{bench}/**/question.jsonl", recursive=True)

            for question_file in list_of_question_files:
                questions = load_questions_jsonl(question_file, release_set, args.livebench_release_option, None)
                bench_questions.extend(questions)

            # Get question IDs for this benchmark
            bench_question_ids = {q['question_id'] for q in bench_questions}

            # Filter df for this specific benchmark
            bench_df = df[df['question_id'].isin(bench_question_ids)]

            if len(bench_df) > 0 and len(bench_questions) > 0:
                print(f"\n=== {bench} ===")
                # Show summary for this specific test
                show_individual_test_summary(args, bench_df, bench_questions, bench, model_filter, valid_question_ids)

    print("\n########## Combined Benchmark Summary ##########")

    # Collect all answer data by model across all benchmarks
    all_answers_by_model = {}

    for bench in args.bench_name:
        answer_files = glob.glob(f"data/{bench}/**/model_answer/*.jsonl", recursive=True)

        for answer_file in answer_files:
            if os.path.exists(answer_file):
                answers = pd.read_json(answer_file, lines=True)

                if len(answers) == 0 or 'model_id' not in answers.columns:
                    continue

                # Filter to only include valid question IDs
                answers = answers[answers['question_id'].isin(valid_question_ids)]

                if len(answers) == 0:
                    continue

                model_id = answers['model_id'].iloc[0]

                # Skip if we're filtering by model list and this model isn't in it
                if model_filter is not None and model_id.lower() not in model_filter:
                    continue

                # Find matching model name from df
                matching_models = [m for m in set(df["model"]) if isinstance(m, str) and m.lower() == model_id.lower()]

                for model in matching_models:
                    if model not in all_answers_by_model:
                        all_answers_by_model[model] = []
                    all_answers_by_model[model].append(answers)

    # Now aggregate the data for each model
    summary_data = {}
    for model, answer_list in all_answers_by_model.items():
        # Combine all answers for this model
        combined_answers = pd.concat(answer_list, ignore_index=True)

        # Filter to only include answers that have judgments (are in df)
        model_question_ids = set(df[df["model"] == model]["question_id"])
        combined_answers = combined_answers[combined_answers['question_id'].isin(model_question_ids)]

        if len(combined_answers) == 0:
            continue

        model_summary = {}

        # Get score for this model across all benchmarks
        model_scores = df[df["model"] == model]["score"]
        if len(model_scores) > 0:
            avg_score = model_scores.mean()
            model_summary['Score'] = round(avg_score, 1)

        # Number of questions answered
        model_summary['Questions'] = len(combined_answers)

        # Total tokens across all questions
        if 'total_output_tokens' in combined_answers.columns:
            valid_tokens = combined_answers[combined_answers['total_output_tokens'] != -1]['total_output_tokens']
            if len(valid_tokens) > 0:
                total_tokens = valid_tokens.sum()
                model_summary['Total Tokens'] = int(total_tokens)

        # Total time across all questions
        total_time = None
        if 'total_inference_time_seconds' in combined_answers.columns:
            valid_times = combined_answers[pd.notna(combined_answers['total_inference_time_seconds'])]['total_inference_time_seconds']
            if len(valid_times) > 0:
                total_time = valid_times.sum()

        # If no inference time available, estimate from timestamps per benchmark
        if total_time is None and 'tstamp' in combined_answers.columns:
            # For API models, we need to sum timing per benchmark, not across all benchmarks
            # Group by benchmark and calculate time for each
            total_time = 0
            for bench in args.bench_name:
                # Get answers for this benchmark
                bench_question_ids = set()
                list_of_question_files = []
                original_question_file = f"data/{bench}/question.jsonl"
                if os.path.exists(original_question_file):
                    list_of_question_files = [original_question_file]
                else:
                    list_of_question_files = glob.glob(f"data/{bench}/**/question.jsonl", recursive=True)

                for question_file in list_of_question_files:
                    questions = load_questions_jsonl(question_file, release_set, args.livebench_release_option, None)
                    bench_question_ids.update([q['question_id'] for q in questions])

                # Filter answers to this benchmark
                bench_answers = combined_answers[combined_answers['question_id'].isin(bench_question_ids)]
                if len(bench_answers) > 0 and 'tstamp' in bench_answers.columns:
                    timestamps = bench_answers['tstamp'].dropna()
                    if len(timestamps) > 1:
                        bench_time = timestamps.max() - timestamps.min()
                        total_time += bench_time

            if total_time == 0:
                total_time = None

        if total_time is not None and total_time > 0:
            model_summary['Total Time (s)'] = round(total_time, 2)

            # Calculate overall average tok/s = total_tokens / total_time
            if 'Total Tokens' in model_summary:
                avg_toks = model_summary['Total Tokens'] / total_time
                model_summary['Avg Tok/s'] = round(avg_toks, 2)

        if model_summary:
            summary_data[model] = model_summary

    if summary_data:
        summary_df = pd.DataFrame(summary_data).T

        # Reorder columns for better readability
        col_order = ['Score', 'Questions', 'Total Tokens', 'Total Time (s)', 'Avg Tok/s']
        available_cols = [col for col in col_order if col in summary_df.columns]
        summary_df = summary_df[available_cols]

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(summary_df)


def display_result_single(args):

    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")
    release_set = set([
        r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option
    ])

    input_files = []
    benchmarks_with_results = []
    for bench in args.bench_name:
        files = (
            glob.glob(f"data/{bench}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
        )
        if args.prompt_testing:
            files += (
                glob.glob(f"prompt_testing/{bench}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
            )
        if files:
            input_files += files
            benchmarks_with_results.append(bench)

    # Filter to only benchmarks that have results
    args.bench_name = benchmarks_with_results

    questions_all = []
    if args.question_source == "huggingface":
        categories = {}
        tasks = {}
        for bench in args.bench_name:
            hf_bench = bench
            # check if bench ends with _{i} for some number i
            number_match = re.match(r'(.*)_\d+$', hf_bench)
            if number_match:
                hf_bench = number_match.group(1)
            bench_cats, bench_tasks = get_categories_tasks(hf_bench)
            categories.update(bench_cats)
            for k, v in bench_tasks.items():
                if k in tasks and isinstance(tasks[k], list):
                    tasks[k].extend(v)
                else:
                    tasks[k] = v
        print(tasks)

        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(categories[category_name], release_set, args.livebench_release_option, task_name, None)
                questions_all.extend(questions)
    elif args.question_source == "jsonl":
        for bench in args.bench_name:
            list_of_question_files = []
            original_question_file = f"data/{bench}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(f"data/{bench}/**/question.jsonl", recursive=True)

            for question_file in list_of_question_files:
                questions = load_questions_jsonl(question_file, release_set, args.livebench_release_option, None)
                questions_all.extend(questions)
    question_id_set = set([q['question_id'] for q in questions_all])

    # Check if we have any judgment files
    if not input_files or len(benchmarks_with_results) == 0:
        return

    df_all = pd.concat((pd.read_json(f, lines=True) for f in input_files), ignore_index=True)
    df = df_all[["model", "score", "task", "category","question_id"]]
    df = df[df["score"] != -1]
    df = df[df['question_id'].isin(question_id_set)]
    df['model'] = df['model'].str.lower()
    df["score"] *= 100
    
    if args.model_list is not None:
        model_list = [get_model_config(x).display_name for x in args.model_list]
        df = df[df["model"].isin([x.lower() for x in model_list])]

    # Keep all models - compute averages based on available data for each model
    # No filtering based on missing judgments

    df.to_csv('df_raw.csv')

    # Always try to show usage information if available (includes scores, tokens, timing)
    calculate_usage(args, df, questions_all, release_set)

    # Save separate CSV files for backwards compatibility
    df_tasks = df[["model", "score", "task"]].groupby(["model", "task"]).mean()
    df_tasks = pd.pivot_table(df_tasks, index=['model'], values="score", columns=["task"], aggfunc="sum")
    df_tasks = df_tasks.round(3).dropna(inplace=False)
    df_tasks.to_csv('all_tasks.csv')

    if not args.prompt_testing:
        df_groups = df[["model", "score", "category", "task"]].groupby(["model", "task", "category"]).mean().groupby(["model","category"]).mean()
        df_groups = pd.pivot_table(df_groups, index=['model'], values="score", columns=["category"], aggfunc="sum")
        df_groups = df_groups.dropna(inplace=False)

        if not args.skip_average_column and len(df_groups.columns) > 1:
            df_groups['average'] = df_groups.mean(axis=1)
            first_col = df_groups.pop('average')
            df_groups.insert(0, 'average', first_col)
            df_groups = df_groups.sort_values(by="average", ascending=False)

        df_groups = df_groups.round(1)
        df_groups.to_csv('all_groups.csv')

        for column in df_groups.columns[1:]:
            max_value = df_groups[column].max()
            df_groups[column] = df_groups[column].apply(lambda x: f'\\textbf{{{x}}}' if x == max_value else x)
        df_groups.to_csv('latex_table.csv', sep='&', lineterminator='\\\\\n', quoting=3,escapechar=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display benchmark results for the provided models")
    parser.add_argument("--bench-name", type=str, default=["live_bench"], nargs="+")
    parser.add_argument(
        "--questions-equivalent", 
        action=argparse.BooleanOptionalAction,
        help="""Use this argument to treat all questions with the same weight. 
        If this argument is not included, all categories have the same weight."""
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--question-source", type=str, default="huggingface", help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will use local jsonl files to permit tweaking or writing custom questions."
    )
    parser.add_argument(
        "--livebench-release-option", 
        type=str, 
        default=max(LIVE_BENCH_RELEASES),
        choices=LIVE_BENCH_RELEASES,
        help="Livebench release to use. Provide a single date option. Will handle excluding deprecated questions for selected release."
    )
    parser.add_argument(
        "--show-average-row",
        default=False,
        help="Show the average score for each task",
        action='store_true'
    )
    parser.add_argument(
        "--ignore-missing-judgments",
        default=False,
        action='store_true',
        help="Ignore missing judgments. Scores will be calculated for only questions that have judgments for all models."
    )
    parser.add_argument(
        "--print-usage",
        default=False,
        action='store_true',
        help="Calculate and display token usage for correct answers"
    )
    parser.add_argument(
        "--verbose",
        default=False,
        help="Display debug information",
        action='store_true'
    )
    parser.add_argument(
        "--skip-average-column",
        default=False,
        help="Skip displaying the average column in results",
        action='store_true'
    )
    parser.add_argument(
        "--prompt-testing",
        default=False,
        help="Use prompt testing results in addition to normal results",
        action="store_true"
    )
    args = parser.parse_args()

    display_result_func = display_result_single

    display_result_func(args)
