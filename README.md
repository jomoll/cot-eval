<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  Evaluating Explanation Faithfulness in Medical Vision–Language
  Models using Multimodal Perturbations
</h1>
</div>
<div align="center">
<h3>
  [Submitted to ML4H 2025]
</h3>
</div>
<div align="center">
</div>

## Setup

### Installation Steps

Follow these steps to set up the environment and get the project running:

```bash
conda env create -f environment.yml
conda activate cot-eval
```

Run inference with modification
```bash
cd scripts
python run_model.py --model-name "google/medgemma-4b-it" --modification vb_hm --leak-correct-answer
```

Run evaluation
```bash
python evaluate.py --evaluation_model "meta-llama/Llama-3.3-70B-Instruct" --model_name "google/medgemma-4b-it" --leak-correct-answer --modification vb_hm
```
