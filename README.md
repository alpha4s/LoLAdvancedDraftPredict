# League of Legends Draft Win Predictor

This project uses a recursive crawler to gather match data from the League of Legends API, trains a PyTorch Wide & Deep Self-Attention Network to predict draft outcomes

## Legal Disclaimer

This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing League of Legends. League of Legends and Riot Games are trademarks or registered trademarks of Riot Games, Inc. League of Legends © Riot Games, Inc.

## Features

- **Data Crawler**: A Breadth-First Search (BFS) crawler that parses match histories using `riotwatcher` and stores high-quality Plat+ solo queue drafts in a local SQLite database.
- **Wide & Deep Attention Network**:
  - **Wide Model**: A linear layer tracking raw champion role-specific win rates.
  - **Deep Model**: Employs embedding layers and a Multi-head Self-Attention (Transformer) block to dynamically learn champion synergies and lane counters.
- **Interactive Web Client**: A local web server featuring a flat, clean dark theme, drag-and-drop drafts, auto-completing champion grids, and real-time win rate updates.
- **Live Recommender**: Select your target slot and pool of champions to receive a sorted recommendation list showing exact win probability deltas.

## Setup

1. **Install Dependencies**:

   ```bash
   pip install torch numpy scikit-learn riotwatcher
   ```

2. **Riot API Key**:
   Get a development key from the [Riot Developer Portal](https://developer.riotgames.com/) and paste it into `data_crawler.py`.

## Usage

1. **Collect Data**:
   Build your SQLite database:

   ```bash
   python data_crawler.py
   ```

2. **Train & Tune Model**:
   Tune hyperparameters and train the network:
   ```bash
   python FFN/tune_nn.py
   ```

## Notes on Patch Sensitivity

The model maps champions by their name strings rather than static stats, so it will not break when Riot releases new balance patches. To keep the counter-picks and win-rate statistics accurate to the current "Meta," run the crawler and re-train the model every few weeks.

## Credits

- **[RiotWatcher](https://github.com/pseudonym117/RiotWatcher)**: Python library for the Riot Games API.
- **Riot Games**: For providing the developer API.
