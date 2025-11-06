import os
import git
import subprocess
from git import Repo
from rich.console import Console
from typing import List

checkout_dict = {
    "astropy/astropy": {
        "5.1": "5f74eacbcc7fff707a44d8eb58adaa514cb7dcb5",
        "5.0": "cdf311e0714e611d48b0a31eb1f0e2cbffab7f23",
        "4.2": "3832210580d516365ddae1a62071001faf94d416",
        "4.3": "298ccb478e6bf092953bca67a3d29dc6c35f6752",
        "3.0": "de88208326dc4cd68be1c3030f4f6d2eddf04520",
        "1.3": "848c8fa21332abd66b44efe3cb48b72377fb32cc",
        "5.2": "362f6df12abf9bd769d4915fabf955c993ea22cf",
        "3.1": "2e89d074b3b2abc2da80e437c93b1d5516a0ca57",
    },
    "django/django": {
        "1.9": "174811c5538c8c0b8f66089b31686e80d2bc7d3b",
        "5.0": "4a72da71001f154ea60906a2f74898d32b7322a7",
        "3.2": "65dfb06a1ab56c238cc80f5e1c31f61210c4577d",
        "2.0": "ddc49820f7716a9e521e8bafda97294065d47b93",
        "4.2": "0fbdb9784da915fce5dcc1fe82bac9b4785749e5",
        "2.1": "3574a6d32fcd88d404b110a8d2204db1dd14a545",
        "3.0": "419a78300f7cd27611196e1e464d50fd0385ff27",
        "3.1": "0668164b4ac93a5be79f5b87fae83c657124d9ab",
        "1.11": "3545e844885608932a692d952c12cd863e2320b5",
        "1.10": "4074fa91452006890a878f0b6a1a25251461cf26",
        "4.1": "647480166bfe7532e8c471fef0146e3a17e6c0c9",
        "4.0": "475cffd1d64c690cdad16ede4d5e81985738ceb4",
        "2.2": "4fc35a9c3efdc9154efce28cb23cb84f8834517e",
    },
    "matplotlib/matplotlib": {
        "3.3": "28289122be81e0bc0a6ee0c4c5b7343a46ce2e4e",
        "3.4": "f93c0a3dcb82feed0262d758626c90d4002685f3",
        "3.5": "de98877e3dc45de8dd441d008f23d88738dc015d",
        "3.2": "c54a5a9b45eff7148e73e9134f206126842307e1",
        "3.7": "0849036fd992a2dd133a0cffc3f84f58ccf1840f",
        "3.0": "d0628598f8d9ec7b0da6b60e7b29be2067b6ea17",
        "3.1": "42259bb9715bbacbbb2abc8005df836f3a7fd080",
        "3.6": "73909bcb408886a22e2b84581d6b9e6d9907c813",
    },
    "mwaskom/seaborn": {
        "0.13": "23860365816440b050e9211e1c395a966de3c403",
        "0.12": "d25872b0fc99dbf7e666a91f59bd4ed125186aa1",
        "0.11": "e8a83c8f12c50eb99bcf32ff83b36bc413ec2e02",
    },
    "pydata/xarray": {
        "2022.03": "d7931f9014a26e712ff5f30c4082cf0261f045d3",
        "0.12": "1c198a191127c601d091213c4b3292a8bb3054e1",
        "2022.09": "087ebbb78668bdf5d2d41c3b2553e3f29ce75be1",
        "2022.06": "50ea159bfd0872635ebf4281e741f3c87f0bef6b",
        "0.19": "df7646182b17d829fe9b2199aebf649ddb2ed480",
        "0.18": "4f1e2d37b662079e830c9672400fabc19b44a376",
        "0.20": "8f42bfd3a5fd0b1a351b535be207ed4771b02c8b",
    },
    "pytest-dev/pytest": {
        "7.2": "572b5657d7ca557593418ce0319fabff88800c73",
        "5.1": "c1361b48f83911aa721b21a4515a5446515642e2",
        "5.0": "c2f762460f4c42547de906d53ea498dd499ea837",
        "7.4": "797b924fc44189d0b9c2ad905410f0bd89461ab7",
        "8.0": "10056865d2a4784934ce043908a0e78d0578f677",
        "4.5": "693c3b7f61d4d32f8927a74f34ce8ac56d63958e",
        "6.0": "634cde9506eb1f48dec3ec77974ee8dc952207c6",
        "4.4": "4ccaa987d47566e3907f2f74167c4ab7997f622f",
        "7.1": "4a8f8ada431974f2837260af3ed36299fd382814",
        "5.2": "f36ea240fe3579f945bf5d6cc41b5e45a572249d",
        "5.4": "678c1a0745f1cf175c442c719906a1f13e496910",
        "5.3": "92767fec5122a14fbf671374c9162e947278339b",
        "7.0": "e2ee3144ed6e241dea8d96215fcdca18b3892551",
        "4.6": "d5843f89d3c008ddcb431adbc335b080a79e617e",
        "6.2": "902739cfc3bbc3379e6ef99c8e250de35f52ecde",
        "6.3": "634312b14a45db8d60d72016e01294284e3a18d4",
    },
    "scikit-learn/scikit-learn": {
        "0.22": "7e85a6d1f038bbb932b36f18d75df6be937ed00d",
        "1.4": "33a1f1690e7a7007633f59b6bee32017f4229864",
        "1.3": "1e8a5b833d1b58f3ab84099c4582239af854b23a",
        "0.21": "7813f7efb5b2012412888b69e73d76f2df2b50b6",
        "0.20": "55bf5d93e5674f13a1134d93a11fd0cd11aabcd1",
    },
    "sympy/sympy": {
        "1.9": "f9a6f50ec0c74d935c50a6e9c9b2cb0469570d91",
        "1.0": "50b81f9f6be151014501ffac44e5dc6b2416938f",
        "1.7": "cffd4e0f86fefd4802349a9f9b19ed70934ea354",
        "1.6": "28b41c73c12b70d6ad9f6e45109a80649c4456da",
        "1.1": "ec9e3c0436fbff934fa84e22bf07f1b3ef5bfac3",
        "1.8": "3ac1464b8840d5f8b618a654f9fbf09c452fe969",
        "1.12": "c6cb7c5602fa48034ab1bd43c2347a7e8488f12e",
        "1.13": "be161798ecc7278ccf3ffa47259e3b5fde280b7d",
        "1.4": "73b3f90093754c5ed1561bd885242330e3583004",
        "1.2": "e53e809176de9aa0fb62e85689f8cdb669d4cacb",
        "1.5": "70381f282f2d9d039da860e391fe51649df2779d",
        "1.11": "9a6104eab0ea7ac191a09c24f3e2d79dcd66bda5",
        "1.10": "fd40404e72921b9e52a5f9582246e4a6cd96c431",
    },
}


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


def prepare_instance(
    task_instance: dict, baseline_temp_dir: str, console: Console
) -> None:
    repo = task_instance.get("repo")
    commit = task_instance.get("base_commit")
    version = task_instance.get("version")
    if not repo or not commit:
        raise ValueError(
            "Both 'repo' and 'base_commit' must be provided in the task instance."
        )

    repo_name = repo.split("/")[-1]
    repo_new_path = os.path.join(baseline_temp_dir, repo_name)

    if os.path.exists(repo_new_path):
        repo = Repo(repo_new_path)
    else:
        repo = git.Repo.clone_from(
            f"https://github.com/{repo}.git", repo_new_path, no_checkout=True
        )
    commit_id = checkout_dict[repo][version]
    console.log(f"[yellow]Checking out {repo} to commit {commit_id}.[/yellow]")
    repo.git.checkout(commit_id)
    packages = package_name_dict.get(repo, [])
    with open(f"/tmp/{repo_name}/package.txt", "w") as f:
        for package in packages[:-1]:
            f.write(f"{package}\n")
        target_package = packages[-1] if packages else ""
        f.write(f"{target_package}=={version}\n")

    console.log(
        f"[yellow]Created package.txt for {repo} with target version {version}.[/yellow]"
    )


def cleanup_instance(
    task_instance: dict,
    console: Console,
    baseline_temp_dir: str,
) -> None:
    repo = task_instance.get("repo")
    if not repo:
        raise ValueError("The 'repo' must be provided in the task instance.")

    repo_name = repo.split("/")[-1]
    repo_path = os.path.join(baseline_temp_dir, repo_name)
    cleanup_cmd = ["rm", "-rf", repo_path]

    try:
        subprocess.run(cleanup_cmd, check=True)
        console.log(f"[yellow]Cleaned up {repo_path}.[/yellow]")
    except subprocess.CalledProcessError as e:
        console.log(f"[red]Error during cleanup: {e}[/red]")
        raise


def run_codamosa(args, task_instances: List[dict], console: Console) -> None:

    os.makedirs(args.baseline_output_path, exist_ok=True)
    os.makedirs(args.baseline_tmp_dir, exist_ok=True)

    for task_instance in task_instances:

        prepare_instance(
            task_instance, baseline_temp_dir=args.baseline_tmp_dir, console=console
        )
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
        output_file = os.path.join(
            args.baseline_output_path,
            f"{args.baseline_output_name}_{task_instance.get('id', 'output')}",
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
