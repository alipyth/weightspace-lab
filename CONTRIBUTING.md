# Contributing

Thanks for your interest in improving WeightSpace Lab.

## Before opening a pull request

1. Use Python 3.10–3.12.
2. Create a virtual environment and install PyTorch plus `requirements.txt`.
3. Run the test suite:

   ```bash
   python -m unittest tests.test_harraf_workspace tests.test_web_api -v
   ```

4. Keep changes compatible with both Harraf 2 and standard Hugging Face folders where relevant.
5. Do not add model checkpoints, access tokens, API keys, user datasets, or private model files to the repository.

## Pull request notes

Describe the behavior changed, the model formats affected, the tests executed, and any limitations. For tensor-editing changes, include validation and reversibility details. For new architecture support, specify the supported task, tokenizer expectations, generation/evaluation path, and export behavior.

## Style

Prefer small, typed Python functions, clear Persian user-facing explanations, and tests for loader, editor, export, and API behavior. Avoid claiming that a statistical signal or weight edit guarantees an improvement in model quality.
