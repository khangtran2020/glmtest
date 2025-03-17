import argparse


def add_general_group(group):
    group.add_argument(
        "--mode",
        type=str,
        default="data",
        help="mode of the program: data, train, test",
    )
    group.add_argument(
        "--name",
        type=str,
        default="testing",
        help="Name to logs to wandb",
    )
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--debug", action="store_true", help="debug mode")
    group.add_argument("--model_debug", action="store_true", help="debug mode")


def add_data_group(group):
    group.add_argument("--data_path", type=str, help="dir path to dataset")
    group.add_argument(
        "--max_pynguin_run_time", type=int, help="time limit for pynguin", default=10
    )
    group.add_argument("--data", type=str, help="name of dataset")
    group.add_argument(
        "--docker-image", type=str, help="docker image to use", default="pynguin_runner"
    )
    group.add_argument("--graph_type", type=str, default="joern", help="graph type")
    group.add_argument("--num_cpu", type=int, default=-1, help="number of cpus to use")
    group.add_argument("--do_crawl", action="store_true", help="crawl the raw data")
    group.add_argument(
        "--do_process_raw", action="store_true", help="process the raw data"
    )
    group.add_argument(
        "--feat_model",
        type=str,
        help="model to extract node features",
        default="Salesforce/codet5p-110m-embedding",
    )
    group.add_argument(
        "--baseline_prompt",
        type=str,
        help="baseline of input prompts",
        default="code",
    )


def add_joern_group(group):
    group.add_argument(
        "--joern_port", type=str, help="port of joern server", default="8080"
    )
    group.add_argument(
        "--joern_path",
        type=str,
        help="path to joern",
        default="./graph/joern/",
    )


def add_training_group(group):
    group.add_argument(
        "--llm_model",
        type=str,
        help="llm model to generate test cases",
        default="Qwen/CodeQwen1.5-7B-Chat",
    )
    group.add_argument(
        "--max_seq_length",
        type=int,
        help="max sequence length",
        default=12000,
    )
    group.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        default=1,
    )
    group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="gradient accumulation steps",
        default=4,
    )
    group.add_argument(
        "--num_gpu",
        type=int,
        help="number of gpus",
        default=1,
    )
    group.add_argument(
        "--output_dir",
        type=str,
        help="output directory to save model",
    )
    group.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="overwrite output directory",
    )
    group.add_argument(
        "--do_train",
        action="store_true",
        help="train the model",
    )
    group.add_argument(
        "--do_eval",
        action="store_true",
        help="evaluate the model",
    )
    group.add_argument(
        "--do_predict",
        action="store_true",
        help="predict the model",
    )
    group.add_argument(
        "--use_lora",
        action="store_true",
        help="train the model",
    )
    group.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate",
        default=5e-5,
    )
    group.add_argument(
        "--rope_theta",
        type=float,
        help="rope theta",
        default=500000.0,
    )
    group.add_argument(
        "--model_max_length",
        type=float,
        help="model_max_length",
        default=16384,
    )
    group.add_argument(
        "--max_grad_norm",
        type=float,
        help="max gradient norm",
        default=1.0,
    )
    group.add_argument(
        "--num_train_epochs",
        type=int,
        help="number of training epochs",
        default=3,
    )
    group.add_argument(
        "--dtype",
        type=str,
        help="model and data type, float32/bfloat16",
        default="bfloat16",
    )
    group.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="resume from checkpoint",
    )
    group.add_argument(
        "--logging_steps",
        type=int,
        help="number of steps to logs",
        default=200,
    )
    group.add_argument(
        "--validating_steps",
        type=int,
        help="number of steps to validate",
        default=200,
    )
    group.add_argument(
        "--run_name",
        type=str,
        help="name of the run for wandb",
        default="testing",
    )
    group.add_argument("--longlora", action="store_true", help="Train with LongLoRA")


def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    joern_group = parser.add_argument_group(title="Joern-related configuration")
    training_group = parser.add_argument_group(title="Training-related configuration")

    add_joern_group(joern_group)
    add_data_group(data_group)
    add_general_group(general_group)
    add_training_group(training_group)
    return parser.parse_args()
