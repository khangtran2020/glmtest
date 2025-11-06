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
        "--graph_sampling", action="store_true", help="crawl the raw data"
    )
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
    group.add_argument(
        "--raw_overwrite", action="store_true", help="overwrite the raw data"
    )
    group.add_argument("--data_fuzz", action="store_true", help="using fuzz data")
    group.add_argument(
        "--repo",
        type=str,
        help="repo to use for training",
        default=None,
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


def add_model_group(group):

    group.add_argument(
        "--gnn_mode",
        type=str,
        help="mode of the program: node, graph",
        default="node",
    )
    group.add_argument(
        "--model_name",
        type=str,
        help="name of the LLM",
        default="qwen2_5-1_5b",
    )
    group.add_argument(
        "--llm_model_name",
        type=str,
        help="name of the LLM",
        default="qwen2_5-1_5b",
    )
    group.add_argument(
        "--in_feats",
        type=int,
        help="number of input features",
        default=772,
    )
    group.add_argument(
        "--n_hidden",
        type=int,
        help="number of hidden features",
        default=64,
    )
    group.add_argument(
        "--n_layers",
        type=int,
        help="number of layers",
        default=3,
    )
    group.add_argument(
        "--num_head",
        type=int,
        help="number of heads",
        default=8,
    )
    group.add_argument(
        "--dropout",
        type=float,
        help="dropout rate",
        default=0.2,
    )
    group.add_argument(
        "--lora_r",
        type=int,
        help="lora rank",
        default=4,
    )
    group.add_argument(
        "--lora_alpha",
        type=int,
        help="lora alpha",
        default=32,
    )
    group.add_argument(
        "--lora_dropout",
        type=float,
        help="lora dropout",
        default=0.1,
    )
    group.add_argument(
        "--lora_target_modules",
        type=str,
        help="lora target modules",
        default=None,
    )
    group.add_argument(
        "--max_num_checkpoint",
        type=int,
        help="maximum number of checkpoints",
        default=2,
    )
    group.add_argument(
        "--model_weight_path", type=str, help="path to the model weight", default=None
    )
    group.add_argument(
        "--checkpoint_path", type=str, help="path to the checkpoint", default=None
    )
    group.add_argument(
        "--continue_training",
        action="store_true",
        help="continue training from checkpoint",
    )
    group.add_argument(
        "--test_on_train",
        action="store_true",
        help="test on train dataset",
    )
    group.add_argument(
        "--fuzz_model",
        action="store_true",
        help="using fuzz model",
    )
    group.add_argument(
        "--fuzzing",
        action="store_true",
        help="Start fuzzing process",
    )
    group.add_argument(
        "--num_samples_per_input",
        type=int,
        help="number of samples to fuzz per input",
        default=10,
    )
    group.add_argument(
        "--start_fuzz_layer_index",
        type=int,
        help="start layer index for fuzzing",
        default=0,
    )
    group.add_argument(
        "--end_fuzz_layer_index",
        type=int,
        help="end layer index for fuzzing",
        default=0,
    )
    group.add_argument(
        "--kl_g_reg",
        type=float,
        help="kl regularization for gaussian",
        default=0.001,
    )
    group.add_argument(
        "--kl_d_reg",
        type=float,
        help="kl regularization for dirichlet",
        default=0.001,
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
        "--temp",
        type=float,
        help="temperature",
        default=0.01,
    )
    group.add_argument(
        "--top_k",
        type=int,
        help="top_k for text generation",
        default=50,
    )
    group.add_argument(
        "--top_p",
        type=float,
        help="top_p for text generation",
        default=0.95,
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
        default=16,
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
        default="./results/",
        help="output directory to save model",
    )
    group.add_argument(
        "--log_dir",
        default="./logs/",
        type=str,
        help="output directory to save model",
    )
    group.add_argument(
        "--gen_dir",
        default="./results/generated/",
        type=str,
        help="output directory to save model",
    )
    group.add_argument(
        "--gen_file_path",
        default="None",
        type=str,
        help="output file",
    )
    group.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="Model directory of the testing model",
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
        "--do_test",
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
        default=1e-3,
    )
    group.add_argument(
        "--rope_theta",
        type=float,
        help="rope theta",
        default=500000.0,
    )
    group.add_argument(
        "--max_new_tokens",
        type=int,
        help="max new tokens to generate",
        default=512,
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
        help="model and data type",
        default="bf16",
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
        "--save_steps",
        type=int,
        help="number of steps to save checkpoint",
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
    group.add_argument(
        "--use_accelerate",
        action="store_true",
        help="Train with accelerate",
    )
    group.add_argument(
        "--only_nvib",
        action="store_true",
        help="Train only the NVIB layers",
    )


def add_testgen_group(group):

    group.add_argument(
        "--module_path",
        type=str,
        help="path to the module to generate test cases",
        default=None,
    )
    group.add_argument(
        "--do_generate",
        action="store_true",
        help="generate test cases for the model",
    )
    group.add_argument(
        "--verifier_model",
        type=str,
        help="model to verify the test cases",
        default=None,
    )
    group.add_argument(
        "--verifier_api_key",
        type=str,
        help="api key for the verifier model",
        default=None,
    )


def add_baseline_group(group):
    group.add_argument(
        "--baseline_type",
        type=str,
        help="type of baseline to use",
        default="prompt_engineer",
    )
    group.add_argument(
        "--baseline_llm_model", type=str, help="baseline llm model", default=None
    )
    group.add_argument(
        "--baseline_api_key",
        type=str,
        help="api key for the baseline llm model",
        default=None,
    )
    group.add_argument(
        "--baseline_output_path",
        type=str,
        help="path to save baseline outputs",
        default=None,
    )
    group.add_argument(
        "--baseline_sif_path",
        type=str,
        help="path to jif file for baseline generation",
        default=None,
    )
    group.add_argument(
        "--baseline_output_name",
        type=str,
        help="file path for baseline generation",
        default=None,
    )
    group.add_argument(
        "--baseline_prompt_type",
        type=str,
        help="type of prompt to use",
        default="zero_shot",
    )
    group.add_argument(
        "--baseline_temp", type=float, help="temperature for baseline", default=0.01
    )
    group.add_argument(
        "--baseline_max_tokens", type=int, help="max tokens for baseline", default=512
    )


def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    joern_group = parser.add_argument_group(title="Joern-related configuration")
    training_group = parser.add_argument_group(title="Training-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    testgen_group = parser.add_argument_group(
        title="Test-case generation configuration"
    )
    baseline_group = parser.add_argument_group(title="Baseline configuration")

    add_joern_group(joern_group)
    add_data_group(data_group)
    add_general_group(general_group)
    add_training_group(training_group)
    add_model_group(model_group)
    add_testgen_group(testgen_group)
    add_baseline_group(baseline_group)

    return parser.parse_args()
