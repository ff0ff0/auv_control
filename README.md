
- Listing the available tasks:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```

- Training:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
    ```

    The task in this repo is:

    ```bash
    python scripts/rsl_rl/train.py --task=Isaac-WarpAUV-Direct-v1 --num_envs 2048
    ```

    Resume training from the latest checkpoint in a run:

    ```bash
    python scripts/rsl_rl/train.py --task=Isaac-WarpAUV-Direct-v1 --resume --load_run <run_folder_name>
    ```

- Playing a trained policy:

    Play the latest checkpoint from a given run:

    ```bash
    python scripts/rsl_rl/play.py --task=Isaac-WarpAUV-Direct-v1 --num_envs 32 --load_run <run_folder_name>
    ```

    Play a specific checkpoint:

    ```bash
    python scripts/rsl_rl/play.py --task=Isaac-WarpAUV-Direct-v1 --num_envs 32 --checkpoint <path_to_checkpoint.pt>
    ```

    Record a demo video while playing:

    ```bash
    python scripts/rsl_rl/play.py --task=Isaac-WarpAUV-Direct-v1 --num_envs 1 --load_run <run_folder_name> --video
    ```

