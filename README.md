# ISL Translator

A lightweight webcam-based Indian Sign Language demonstrator for practical
banking and hospital communication. MediaPipe extracts 21 hand landmarks per
frame; a compact LSTM recognizes complete 30-frame phrases.

## Setup in VS Code

Open this folder in VS Code, select the `venv311` Python interpreter, then run:

```powershell
.\venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the pipeline

1. Collect consistent samples for each complete phrase. Record at least 20
   sequences per phrase, preferably from 3-5 people. The current scope is:
   `BANK_BALANCE`, `CASH_WITHDRAWAL`, `MONEY_TRANSFER`, `HOSPITAL_DOCTOR`,
   `MEDICINE`, and `EMERGENCY`.

   ```powershell
   python src/collect_gestures.py BANK_BALANCE --sequences 20
   python src/collect_gestures.py EMERGENCY --sequences 20
   ```

2. Train after collecting two or more labels:

   ```powershell
   python src/train_model.py --epochs 100
   ```

3. Start recognition:

   ```powershell
   python src/live_translate.py
   ```

Press `q` to close either webcam window. The model and its label mapping are
saved under `models/`. Do not change the sequence length after collecting data.

## Static service-counter phrases

`dataset/common_phrases/` supplies MediaPipe-Holistic landmark samples for
static signs. The static classifier uses only the two hand-landmark portions
and trains these service-counter phrases: hello, namaste, please, yes, no,
sorry, thanks, understand, water, and food.

```powershell
python src/train_static_phrases.py
python src/live_static_phrases.py
```

This stays separate from the dynamic 30-frame banking/hospital phrase model.

## Checks

```powershell
python -m pytest
```
