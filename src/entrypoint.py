import subprocess
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import List


@dataclass
class TaskArguments:
    command: str = field(metadata={"help": "The command to run: 'train' or 'inference'"})


@dataclass
class CloudArguments:
    num_processes: int = field(default=4, metadata={"help": "Number of GPU processes"})
    num_cpu_threads_per_process: int = field(default=24, metadata={"help": "Number of CPU cores"})


def run_command(command_list: List[str]):
    print(f"Running command: {' '.join(command_list)}")
    subprocess.run(command_list, check=True)


def main():
    parser = HfArgumentParser((TaskArguments,))
    task_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if task_args.command == "inference":
        parser = HfArgumentParser((CloudArguments,))
        cloud_args, eval_args = parser.parse_args_into_dataclasses(args=remaining_args, return_remaining_strings=True)
        print("--- Starting Inference with Accelerate ---")

        cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_machines", "1",
            "--num_processes", str(cloud_args.num_processes),
            "--num_cpu_threads_per_process", str(cloud_args.num_cpu_threads_per_process),
            "/app/train/inference.py"
        ]

        cmd.extend(eval_args)
        run_command(cmd)

    elif task_args.command == "train":
        parser = HfArgumentParser((CloudArguments,))
        cloud_args, train_args = parser.parse_args_into_dataclasses(args=remaining_args, return_remaining_strings=True)
        print("--- Starting Training with Accelerate ---")

        cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--num_machines", "1",
            "--num_processes", str(cloud_args.num_processes),
            "--num_cpu_threads_per_process", str(cloud_args.num_cpu_threads_per_process),
            "/app/train/finetune.py"
        ]

        cmd.extend(train_args)
        run_command(cmd)

    else:
        raise ValueError(f"Unknown command: {task_args.command}")


if __name__ == "__main__":
    main()
