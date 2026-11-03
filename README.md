# WeightSpace Lab

**A Persian-first web workbench for inspecting, analyzing, editing, and benchmarking open-weight AI models.**

WeightSpace Lab helps researchers and developers explore neural-network checkpoints at the tensor level, understand common model components, test controlled weight edits, and compare model behavior before and after an experiment.

The application runs locally and opens in a browser. It supports Hugging Face Hub repositories, local Transformers model folders, standalone weight checkpoints, and the custom Harraf 2 architecture.

[راهنمای فارسی](README_FA.md) · [Project website](https://jahaniwww.com) · [Telegram channel](https://t.me/Tarfandoon)

## Features

- Load a model directly from the **Hugging Face Hub** using an `owner/model` repository ID.
- Load a local Hugging Face folder with `config.json`, tokenizer assets, single-file or sharded weights.
- Inspect PyTorch `.pt`, `.pth`, `.ckpt`, `.bin`, and `.safetensors` checkpoints.
- Automatically summarize model architecture, likely task, configuration, tensor names, shapes, dtypes, and parameter roles.
- Review tensor statistics, distributions, normalization parameters, embeddings, attention projections, MLP weights, and other common components.
- Run structural diagnostics and activation probes on supported architectures.
- Ask an OpenAI-compatible analysis model (including Ollama) for a structured, model-wide investigation and a ranked experiment plan.
- Apply reversible edits such as scaling, slice scaling, controlled noise, and clamping.
- Keep snapshots, undo/redo edits, export a separate checkpoint, and inspect exact weight differences.
- Compare supported language models with fixed prompts and surface-level output metrics.
- Use a Persian RTL dashboard with light and dark themes and Vazirmatn typography.

## Supported model sources

| Source | What is supported |
|---|---|
| Hugging Face Hub | Public repositories and gated/private repositories when an access token is supplied; files are downloaded locally. |
| Local Hugging Face folder | `config.json`, tokenizer files, `model.safetensors`, `pytorch_model.bin`, and common sharded variants with index files. |
| PyTorch checkpoint | `.pt`, `.pth`, `.ckpt`, or `.bin` files containing a tensor `state_dict` or a supported checkpoint dictionary. |
| SafeTensors | Single-file `.safetensors` checkpoints, with optional model configuration alongside. |
| Harraf 2 | Native loading, generation, activation analysis, editing, and before/after evaluation using its embedded configuration and companion assets. |

### Execution and evaluation support

Standard Hugging Face causal language models and sequence-to-sequence models can use Transformers for text generation and before/after comparisons. Other model types (for example, masked language models, classifiers, or vision encoders) can be inspected and edited when Transformers can load them, but text generation is not assumed to be meaningful for every task.

A raw checkpoint without a usable configuration or architecture adapter can still be inspected and edited as tensors. It cannot be given a reliable behavioral score until the matching architecture, tokenizer, input format, and task-specific evaluation are available.

## Quick start — Windows

Requirements: Python 3.10–3.12 and an internet connection for the initial dependency installation or model download.

1. Download or clone this repository.
2. Run `run.bat`.
3. The launcher creates a virtual environment, installs CPU PyTorch and the project requirements, then opens the local dashboard.
4. The dashboard is available at `http://127.0.0.1:8765`.
5. Load a model from a local folder, enter a Hugging Face repository ID, or choose a checkpoint directory.

Keep the terminal window open while using the application. Closing it stops the local server.

## Quick start — Linux

```bash
chmod +x run.sh
./run.sh
```

The script creates `.venv`, installs CPU PyTorch and the Python requirements, and starts the dashboard at `http://127.0.0.1:8765`.

## Manual installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in your browser.

## Loading from Hugging Face

Enter a repository ID such as `Qwen/Qwen2.5-0.5B-Instruct` in the Hugging Face Hub section. A revision can be specified; the default is the repository's default revision. For gated or private repositories, provide a Hugging Face access token for that request.

The application uses `huggingface_hub` to download the repository and `transformers` to load the model. `trust_remote_code` is disabled. Models that require custom Python code from their repository are intentionally not executed by this application.

For a local model, select the complete folder. Sharded checkpoints are handled by Transformers when the required index and shard files are present.

## Model analysis and weight editing

The model profile is inferred from its configuration, declared architecture, metadata, and parameter naming patterns. Tensor descriptions are based on common architecture conventions; they are not proof of a model's learned behavior.

The analysis provider accepts an OpenAI-compatible API endpoint. For example, an Ollama endpoint can be configured as `http://localhost:11434/v1`. The language model receives a structured dossier containing model metadata, tensor inventory, shape and dtype information, role descriptions, statistical summaries, diagnostic findings, and available behavioral samples. Raw full-size parameter matrices are not sent to the analysis provider.

The edit history is reversible. Each batch is validated before application and recorded in a snapshot. The original model files are not overwritten; exports are written to a new directory or downloaded as a separate archive.

## Evaluation

For supported text-generation models, the application can run the same prompt set against the original checkpoint and the edited model. It reports output text and surface-level measures such as repeated n-gram ratio, n-gram diversity, response length, and output entropy where available.

These metrics do not establish factual accuracy, reasoning ability, or overall model quality. For serious evaluation, use task-specific datasets with reference answers and add human or domain-specific scoring.

## Safety and limitations

- Browser folder uploads are streamed to disk; the current upload limit is 64 GB per request.

- The server binds to `127.0.0.1` by default. Do not expose it to a public network without adding authentication, authorization, request limits, and appropriate deployment controls.
- PyTorch checkpoint loading uses `weights_only=True`. Pickle-based arbitrary object loading is not enabled.
- Hugging Face `trust_remote_code` is disabled.
- Large models can require substantial RAM and disk space. CPU inference may be slow.
- Quantized formats such as GGUF, GPTQ, and AWQ are not currently supported by the tensor editor.
- LoRA/PEFT adapter editing is not currently supported.
- Statistical anomalies are indicators for investigation, not causal diagnoses. Weight edits are experiments and do not guarantee an improvement.

## Tests

```bash
python -m unittest tests.test_harraf_workspace tests.test_web_api -v
```

The Hugging Face integration test runs when `transformers` is installed. The remaining tests cover the Harraf 2 adapter, generic checkpoint inspection, editing, undo/redo, model export, and web API behavior.

## Project information

- **Project:** WeightSpace Lab
- **Developer:** Ali Jahani (علی جهانی)
- **Website:** https://jahaniwww.com
- **Telegram:** https://t.me/Tarfandoon
- **Interface language:** Persian (RTL), with Vazirmatn typography
