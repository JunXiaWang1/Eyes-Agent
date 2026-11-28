# Eyes-Agent

Eyes-Agent is a research demo for eye-level glaucoma prediction and visual question answering (VQA). You can run it on a single eye or on both eyes of a case. The two prediction modes first score photographs locally, then ask GPT-5 to combine the scores with the available clinical history. VQA sends the photographs, questions, and optional clinical text to the OpenAI API and returns an answer for each question.

| Mode | Images used | Clinical text | Result per eye |
| --- | --- | --- | --- |
| `standard` | Fundus photographs, anterior-segment photographs, or both | `clinical.txt` | Visual model scores and a fused glaucoma probability |
| `smartphone` | Anterior-segment photographs processed by the smartphone pipeline | `clinical_smartphone.txt` | Anterior-segment model score and a fused glaucoma probability |
| `vqa` | Available fundus and anterior-segment photographs | `clinical.txt` | Answers to the questions in `questions.json` |

## Setup

Run the commands below from the directory containing `demo.py` and `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

Set the environment variable `OPENAI_API_KEY` in the terminal where you will run the demo. Use a key with access to the model for your chosen mode. Fusion uses `gpt-5`. VQA defaults to `gpt-5.2` and accepts a different image-capable model with structured output support via `--vqa-model`.

| Shell | Set the key for the current session |
| --- | --- |
| Bash | `export OPENAI_API_KEY="your-api-key"` |
| Windows PowerShell | `$env:OPENAI_API_KEY = "your-api-key"` |

For fusion, place the study's fine-tuned classifiers at the paths below. The demo package contains the model definitions. Supply the checkpoint for each image type you use.

| Checkpoint path | Used by |
| --- | --- |
| `weights/fundus.pth` | Standard fusion when fundus images are present |
| `weights/anterior_segment.pth` | Standard fusion when anterior-segment images are present, and smartphone fusion |

Download both checkpoints from [Eyes-Agent-Weights](https://huggingface.co/wjx1415413364/Eyes-Agent-Weights/tree/main) and place them in the `weights/` directory.

VQA runs directly through the API and requires no local checkpoint. For a first run without weights, use the VQA command below.

## Run an example

The bundled `demo_cases/case_001_R` example contains images of a right eye. Run the command for the mode you want.

```bash
# Visual question answering
python demo.py --mode vqa --case demo_cases/case_001_R --side R

# Standard fusion
python demo.py --mode standard --case demo_cases/case_001_R --side R

# Smartphone fusion
python demo.py --mode smartphone --case demo_cases/case_001_R --side R
```

The smartphone command demonstrates the smartphone pipeline using the example's anterior-segment folder. For your own smartphone case, put the phone photographs in that folder. For either fusion mode, add `--device cpu` or `--device cuda` to choose a device explicitly. By default, the classifiers use CUDA when available and CPU otherwise.

## Prepare a case

Use the following layout for a single eye. Paths in the table are relative to the case directory.

| Path | Contents |
| --- | --- |
| `fundus/` | Fundus photographs of this eye |
| `anterior_segment/` | Anterior-segment photographs of this eye |
| `clinical.txt` | Optional clinical history for standard fusion and VQA |
| `clinical_smartphone.txt` | Optional clinical history for smartphone fusion. Omit fundus findings, including optic disc descriptions and cup-to-disc ratio. If the file is absent, the mode proceeds with the image model score alone. |
| `cdr.txt` | Optional precomputed vertical cup-to-disc ratio, written as a single number from 0 to 1. Standard fusion reads it when fundus images are present. |
| `questions.json` | VQA question file. Keep this exact filename and follow the format in the bundled example. The reader omits reference answers from the API request. |

Use de-identified inputs suitable for transmission to the API. Supported image extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`. For the bundled single-eye folders, pass `--side L` or `--side R` explicitly. For a bilateral case, put images and eye-specific files (`cdr.txt`, `questions.json`) under `L/` and `R/`. Keep the clinical text files at the case root and omit `--side` to process both eyes.

## Read the output

The demo prints JSON to the terminal. The example below illustrates the standard fusion format with **fictional scores**.

```json
{
  "mode": "standard",
  "visual_results": {
    "R": {
      "fundus_prob": 0.82,
      "anterior_prob": 0.65
    }
  },
  "fusion_probabilities": {
    "R": 0.781
  }
}
```

For each eye, `visual_results` lists the available model scores. When several photos of one modality are present, the reported score is the arithmetic mean of the individual probabilities. Class 1 denotes glaucoma and class 0 denotes non-glaucoma. Read `fusion_probabilities.R` or `fusion_probabilities.L` for the final estimated probability. Smartphone mode uses the same output structure with `"mode": "smartphone"` and a single visual score, `anterior_prob`.

In VQA mode, read `results.R.answers` or `results.L.answers`. Each item contains a question ID and an answer, for example `{"question_id": "1-1", "answer": "Yes"}`. Binary answers use `Yes` or `No`. The study's categorical treatment question uses `surgery` or `medication` as benchmark labels.
