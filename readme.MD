# WiM-RL: Reinforcement Learning Enhanced Window in Mind

A reinforcement learning approach to improve margin generation in the Window in Mind (WiM) framework for long document question answering.

## Overview

This project enhances the WiM (Window in Mind) approach by incorporating reinforcement learning to generate better margins (extractive summaries) from document segments. The system uses PPO (Proximal Policy Optimization) to train a margin generator that produces more relevant and informative summaries for answering queries.

## Features

- **RL-Enhanced Margin Generation**: Uses reinforcement learning to improve the quality of extracted margins
- **PPO Training**: Implements Proximal Policy Optimization for stable training
- **Reward-Based Learning**: Custom reward model that evaluates margin quality based on relevance and information density
- **Flexible Architecture**: Supports various transformer models (Llama, H2O-Danube, etc.)
- **Quantized Model Support**: Includes support for quantized models for efficient inference

## Project Structure

```
Wim_RL/
├── wim_new/                    # Latest implementation
│   ├── WiM_inference.py        # Main WiM-RL inference pipeline
│   ├── RL_margin_generation.py # RL margin generator and reward model
│   ├── wim.py                  # Core WiM implementation
│   └── main.py                 # Entry point script
├── wim_old/                    # Previous implementation versions
├── examples/                   # Example datasets (BabiLong)
│   ├── babilong_8k.json
│   ├── babilong_16k.json
│   └── ...
├── output/                     # Training outputs and results
├── templates/                  # Prompt templates
├── kaggle.ipynb              # Kaggle notebook implementation
├── kaggle_quantized_model.ipynb # Quantized model experiments
└── RL_train.ipynb            # RL training notebook
```

## Key Components

### 1. RLMarginGenerator

- **Location**: [`wim_new/RL_margin_generation.py`](wim_new/RL_margin_generation.py)
- Implements PPO-based training for margin generation
- Includes policy model, reference model, and reward computation
- Supports KL divergence regularization

### 2. WIMRLInference

- **Location**: [`wim_new/WiM_inference.py`](wim_new/WiM_inference.py)
- Extended WiM inference class with RL-enhanced margin generation
- Handles both standard WiM and RL-based approaches
- Supports training and inference modes

### 3. MarginRewardModel

- **Location**: [`wim_new/RL_margin_generation.py`](wim_new/RL_margin_generation.py)
- Computes rewards for generated margins
- Evaluates relevance, coherence, and information density
- Includes classifier agreement bonus

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Wim_RL

# Install dependencies
pip install torch transformers tqdm numpy nltk tiktoken
pip install datasets accelerate bitsandbytes  # For quantization support

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Basic Usage

```python
from wim_new.WiM_inference import run_wim_rl

# Run WiM with RL-enhanced margin generation
final_answer, positive_margins = run_wim_rl(
    model_id="h2oai/h2o-danube3-500m-base",
    model_id_rl="h2oai/h2o-danube3-500m-base",
    input_document="Your long document text...",
    query="Your question here",
    use_rl_generator=True,
    train_rl_generator=False,  # Set True to train
    num_episodes=10
)
```

### Training RL Model

```python
# Train the RL margin generator
final_answer, positive_margins = run_wim_rl(
    model_id="h2oai/h2o-danube3-500m-base",
    model_id_rl="h2oai/h2o-danube3-500m-base",
    input_document="Training document...",
    query="Training query",
    use_rl_generator=True,
    train_rl_generator=True,
    num_episodes=5,
    output_model_dir="./trained_model"
)
```

### Configuration

The [`RLConfig`](wim_new/RL_margin_generation.py) dataclass allows customization of RL parameters:

```python
from wim_new.RL_margin_generation import RLConfig

config = RLConfig(
    learning_rate=5e-5,
    kl_coef=0.05,
    discount_factor=0.99,
    ppo_epochs=4,
    clip_param=0.2
)
```

## Notebooks

- **[`kaggle.ipynb`](kaggle.ipynb)**: Main implementation notebook
- **[`kaggle_quantized_model.ipynb`](kaggle_quantized_model.ipynb)**: Quantized model experiments
- **[`RL_train.ipynb`](RL_train.ipynb)**: Focused RL training notebook

## Examples

The [`examples/`](examples/) directory contains BabiLong dataset samples:

- [`babilong_8k.json`](examples/babilong_8k.json): 8K token examples
- [`babilong_16k.json`](examples/babilong_16k.json): 16K token examples
- [`babilong_32k.json`](examples/babilong_32k.json): 32K token examples
- [`babilong_64k.json`](examples/babilong_64k.json): 64K token examples

## Model Support

The framework supports various transformer models:

- **H2O-Danube**: `h2oai/h2o-danube3-500m-base`
- **Llama**: `meta-llama/Llama-3.2-1B-Instruct`
- **Custom fine-tuned models**: `shubvhamgore18218/WiM_llama_full_dataset`

## Training Process

1. **Segment Generation**: Document is chunked into segments
2. **Margin Generation**: RL model generates extractive summaries
3. **Classification**: Margins are classified for relevance
4. **Reward Computation**: Quality scores based on relevance and coherence
5. **Policy Update**: PPO updates using computed rewards
6. **KL Regularization**: Prevents divergence from reference model

## Output

Training produces:

- **Average Reward**: Quality metric across episodes
- **KL Divergence**: Policy divergence from reference
- **Relevance Rate**: Percentage of relevant margins
- **Model Checkpoints**: Saved in [`output/`](output/) directory

## Results

Sample training output from [`output/output.txt`](output/output.txt):

```
Episode 1 Summary:
  Average reward: 0.5223
  New best average reward: 0.5223 (Improvement)

Episode 2 Summary:
  Average reward: 0.5260
  New best average reward: 0.5260 (Improvement)
```
