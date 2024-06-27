# Monte Carlo Tree Search with Self-Refine

This repository contains an implementation of Monte Carlo Tree Search (MCTS) with self-refinement, using a locally hosted LLaMA instance for generating answers. This implementation specifically focuses on handling `gsm8k` and `MATH` datasets using the Hugging Face `datasets` library.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/naivoder/MCTSr.git
    cd MCTSr/
    ```

2. **Set up a virtual environment and activate it:**

    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download and set up the LLaMA model:**

    Follow the instructions provided by Hugging Face to download and set up the LLaMA model.

## Usage

1. **Modify the `MODEL_NAME` and `DATA_NAME` variables in `main.py` as needed:**

    ```python
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    DATA_NAME = "gsm8k-rs-mistral7B"
    ```

2. **Run the main script:**

    ```sh
    python main.py
    ```
