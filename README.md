# League of Legends Draft Win Predictor

A draft-only machine-learning project that estimates the Blue and Red teams' win probabilities from their selected champions. It collects recent ranked matches with the Riot Games API, trains a small PyTorch wide-and-deep model, exports it to ONNX, and runs predictions locally in the browser.

## What It Does

- Crawls ranked Summoner's Rift matches and stores drafts in SQLite.
- Learns champion and champion-role effects from match outcomes.
- Uses simple draft statistics for role frequency, damage profile, off-meta picks, and selected lane matchups for top, mid, and support.
- Supports incomplete drafts by training on randomly masked champion selections.
- Exports a self-contained ONNX model that accepts ten champion IDs.
- Provides a drag-and-drop draft interface with live win probabilities and champion recommendations.

## Model

The model has two paths:

- **Wide path:** learns a direct contribution for each champion in each role.
- **Deep path:** converts the ten selected champions into learned vectors, combines them with precomputed draft statistics, and passes them through a small `64 -> 32 -> 1` neural network using LayerNorm and GELU.

The precomputed statistics are stored as frozen lookup tables inside the model. Because those lookups are included in the ONNX export, the browser only needs to send champion IDs and cannot disagree with Python about feature calculations.

This model intentionally uses draft information only. It does not know player skill, team communication, item choices, or events that occur during the match, so its accuracy should be interpreted as draft-level signal rather than a complete game prediction.

## Setup

Python 3.10 or newer is recommended.

```bash
pip install torch numpy pandas scikit-learn riotwatcher onnx
```

Get a development API key from the [Riot Developer Portal](https://developer.riotgames.com/), then set it as an environment variable in the terminal that will run the crawler.

PowerShell:

```powershell
$env:RIOT_API_KEY = "RGAPI-your-key-here"
```


The crawler reads this variable at startup, so the key does not need to be stored in a project file. Development keys expire every 24 hours and must be replaced when they expire. The included routing defaults target North America (`americas`/`na1`).

## Usage

### 1. Collect matches

```bash
python data_crawler.py
```

The crawler creates `league_data.db`, resumes from previously processed players, and collects champion drafts plus the participant damage statistics used to create champion damage profiles. The current training pipeline keeps only matches whose game version begins with `16` so older seasons do not influence the model.

### 2. Train and export the model

```bash
python FFN/train_nn.py
```

Training performs an 80/20 train-validation split, builds feature tables from the training split only, trains the model, reports validation accuracy, saves the PyTorch weights, and exports the browser model:

- `FFN/model_nn.pth`
- `FFN/model_nn_metadata.json`
- `FFN/model_nn.onnx`

### 3. Run the interface

From the project directory, start a local static server:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000). A local server is required because the interface loads JavaScript modules, JSON metadata, and the ONNX model as separate files.

## Evaluation

The final result should be compared with a majority-class baseline, such as always predicting the side that wins most frequently in the validation data. Report both numbers rather than presenting accuracy without context.

The current model was trained and evaluated using approximately 76,000 season-16 matches. Feature statistics were calculated from the training split only.

```text
Validation matches: 15,261
Majority baseline:  50.77%
Draft model:        52.45%
Improvement:        +1.68 percentage points
```

The reported number is held-out validation accuracy. Validation loss is used to select the exported checkpoint, so this is not presented as accuracy on a separate untouched test set. Incomplete drafts remain close to chance because they contain less information; the headline result uses complete 10-champion drafts.

## Project Layout

```text
data_crawler.py          Riot API crawler and SQLite storage
feature_engineering.py   Training-only aggregate feature calculations
model.py                 PyTorch wide-and-deep model
FFN/train_nn.py          Training, validation, checkpointing, and ONNX export
FFN/predict_nn.py        Python prediction helper
static/js/               Browser interface and ONNX inference
index.html               Draft interface
config.py                Shared paths and model configuration
```

## Patch Sensitivity

Champion balance, builds, and matchups change between patches. The model will continue recognizing existing champion names, but its learned relationships become stale over time. Recollect recent matches and retrain when the model should represent a newer patch.

## Legal Disclaimer

This project is not endorsed by Riot Games and does not reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games and League of Legends are trademarks or registered trademarks of Riot Games, Inc. League of Legends &copy; Riot Games, Inc.

## Credits

- [RiotWatcher](https://github.com/pseudonym117/RiotWatcher) for the Python Riot Games API wrapper.
- Riot Games for providing the developer API.
