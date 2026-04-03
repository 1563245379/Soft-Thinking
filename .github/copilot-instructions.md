# Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space

## Project Overview
This repository contains the official implementation of the paper ["Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space"](https://arxiv.org/abs/2505.15778).

### Key Technologies
- **Python** 
- **PyTorch** & **Transformers**
- **SGLang** (Customized version included in `sglang_soft_thinking_pkg`)
- **FlashAttention**
- **Docker**

## Building and Running

### Environment Setup

A local Conda environment can be set up via the provided scripts:
```bash
# Setup conda environment and install dependencies
bash configure.sh
```

Before running any Python command in this workspace, first activate the required Conda environment:
```bash
source /workspace/miniconda/bin/activate st
```

### Running Inference and Evaluation
The `scripts/` directory contains bash scripts for running baselines and the "Soft Thinking" implementations on various models (e.g., QwQ-32B). 
- **Baseline:**
  ```bash
  bash scripts/baseline/qwq32b.sh
  ```
- **Soft Thinking (Math Evaluation):**
  ```bash
  bash scripts/st/qwq32b_st_math.sh
  ```

All inferences are typically run using the `run_sglang_softthinking.py` script. *Note: An OpenAI API key is required when using the LLM judge for evaluation.*
```bash
export OPENAI_API_KEY="your-api-key"
```

## Development Conventions
- **Licensing Structure:** 
  - The customized SGLang directory (`sglang_soft_thinking_pkg/`) is licensed under the Apache License 2.0. Changes from SGLang v0.4.6.post1 are documented in `change_0.4.6.post1.diff`.
  - Original code in this repository is licensed under the MIT License.
- **Hardware Constraints:** Soft thinking yields suboptimal results on smaller models (<= 14B) due to limited hidden sizes causing noise during probability weighting.
- **Evaluation:** When evaluating on coding benchmarks (HumanEval, MBPP, LiveCodeBench), run inference first without `--reeval`, then run again with `--reeval` to bypass multiprocessing bugs.