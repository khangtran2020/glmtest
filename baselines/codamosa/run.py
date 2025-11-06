import os
import subprocess
from rich.console import Console
from typing import List

package_name_dict = {
    "astropy/astropy": [
        "numpy",
        "pyerfa",
        "astropy-iers-data",
        "pyyaml",
        "packaging",
        "astropy",
    ],
    "django/django": [
        "asgiref",
        "sqlparse",
        "tzdata",  # platform-specific on Windows
        "Django",
    ],
    "matplotlib/matplotlib": [
        "contourpy",
        "cycler",
        "fonttools",
        "kiwisolver",
        "numpy",
        "packaging",
        "pillow",
        "pyparsing",
        "python-dateutil",
        "matplotlib",
    ],
    "mwaskom/seaborn": [
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",  # optional for advanced stats
        "statsmodels",  # optional for advanced stats
        "seaborn",
    ],
    "pydata/xarray": [
        "numpy",
        "packaging",
        "pandas",
        "xarray",
    ],
    "pytest-dev/pytest": [
        "iniconfig",
        "packaging",
        "pluggy",
        "pygments",
        "tomli",
        "colorama",  # used on Windows consoles
        "exceptiongroup",  # used for grouped exceptions
        "pytest",
    ],
    "scikit-learn/scikit-learn": [
        "numpy",
        "scipy",
        "joblib",
        "threadpoolctl",
        "scikit-learn",
    ],
    "sympy/sympy": [
        "mpmath",
        "sympy",
    ],
}


def prepare_instance(task_instance: dict, console: Console) -> None:
    repo = task_instance.get("repo")
    commit = task_instance.get("base_commit")
    version = task_instance.get("version")
    if not repo or not commit:
        raise ValueError(
            "Both 'repo' and 'base_commit' must be provided in the task instance."
        )

    repo_name = repo.split("/")[-1]
    clone_cmd = ["git", "clone", repo, f"/tmp/{repo_name}"]
    checkout_cmd = ["git", "checkout", commit, f"/tmp/{repo_name}"]

    try:
        subprocess.run(clone_cmd, check=True)
        subprocess.run(checkout_cmd, check=True)
        console.log(
            f"[yellow]Cloned and checked out {repo} at commit {commit}.[/yellow]"
        )
    except subprocess.CalledProcessError as e:
        console.log(f"[red]Error during git operations: {e}[/red]")
        raise

    # Create package.txt file to run codamosa
    packages = package_name_dict.get(repo, [])
    with open(f"/tmp/{repo_name}/package.txt", "w") as f:
        for package in packages[:-1]:
            f.write(f"{package}\n")
        target_package = packages[-1] if packages else ""
        f.write(f"{target_package}=={version}\n")

    console.log(
        f"[yellow]Created package.txt for {repo} with target version {version}.[/yellow]"
    )


def cleanup_instance(task_instance: dict, console: Console) -> None:
    repo = task_instance.get("repo")
    if not repo:
        raise ValueError("The 'repo' must be provided in the task instance.")

    repo_name = repo.split("/")[-1]
    cleanup_cmd = ["rm", "-rf", f"/tmp/{repo_name}"]

    try:
        subprocess.run(cleanup_cmd, check=True)
        console.log(f"[yellow]Cleaned up /tmp/{repo_name}.[/yellow]")
    except subprocess.CalledProcessError as e:
        console.log(f"[red]Error during cleanup: {e}[/red]")
        raise


def run_codamosa(args, task_instances: List[dict], console: Console) -> None:

    for task_instance in task_instances:

        prepare_instance(task_instance, console)
        repo = task_instance.get("repo")
        if not repo:
            raise ValueError("The 'repo' must be provided in the task instance.")

        repo_name = repo.split("/")[-1]
        """
        Command to run is something like:
        apptainer run \
            --bind $TEST_BASE/test-apps/flutils:/input:ro \
            --bind /tmp/flutils-out:/output \
            --bind $TEST_BASE/test-apps/flutils:/package:ro \
            Sif file \
            --project_path /input \
            --module-name flutils.packages \
            --output-path /output \
            --report-dir /output \
            --maximum_search_time 120 \
            --output_variables TargetModule,CoverageTimeline \
            --coverage_metrics BRANCH,LINE \
            --assertion-generation NONE \
            --algorithm CODAMOSA \
            -v \
            --include-partially-parsable True \
            --allow-expandable-cluster True \
            --uninterpreted_statements ONLY \
            --temperature 0.8 \
            --model_name code-davinci-002 \
            --authorization-key "$AUTH_KEY" \
            --model_base_url "<BASE_URL>" \
            --model_relative_url "<RELATIVE_URL>"
        """
        output_file = os.path.join(args.baseline_output_path, args.baseline_output_name)
        if "claude" in args.baseline_llm_model.lower():
            url = "https://api.anthropic.com/v1/messages"
        elif (
            "o3" in args.baseline_llm_model.lower()
            or "gpt" in args.baseline_llm_model.lower()
        ):
            url = "https://api.openai.com/v1/chat/completions"

        codamosa_cmd = [
            "apptainer",
            "run",
            "--bind",
            f"/tmp/{repo_name}:/input:ro",
            "--bind",
            f"{output_file}:/output",
            "--bind",
            f"/tmp/{repo_name}:/package:ro",
            args.baseline_sif_path,
            "--project_path",
            "/input",
            "--module-name",
            task_instance.get("code_file", ""),
            "--output-path",
            "/output",
            "--report-dir",
            "/output",
            "--maximum_search_time",
            "600",  # 10 minutes
            "--output_variables",
            "TargetModule,CoverageTimeline",
            "--coverage_metrics",
            "BRANCH,LINE",
            "--assertion-generation",
            "NONE",
            "--algorithm",
            "CODAMOSA",
            "-v",
            "--include-partially-parsable",
            "True",
            "--allow-expandable-cluster",
            "True",
            "--uninterpreted_statements",
            "ONLY",
            "--temperature",
            str(args.baseline_temp),
            "--model_name",
            args.baseline_llm_model,
            "--authorization-key",
            args.baseline_api_key,
            "--model_base_url",
            url,
        ]

        try:
            subprocess.run(codamosa_cmd, check=True)
            console.log(f"[yellow]Ran Codamosa on /tmp/{repo_name}.[/yellow]")
        except subprocess.CalledProcessError as e:
            console.log(f"[red]Error during Codamosa execution: {e}[/red]")
            raise

        cleanup_instance(task_instance, console)
