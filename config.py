import os

# Project Root Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File & Database Paths
DB_PATH = os.path.join(BASE_DIR, 'league_data.db')
CHAMPIONS_PATH = os.path.join(BASE_DIR, 'champions.json')
FEATURE_CACHE_PATH = os.path.join(BASE_DIR, 'feature_matrices.json')

MODEL_DIR = os.path.join(BASE_DIR, 'FFN')
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'model_nn.pth')
MODEL_META_PATH = os.path.join(MODEL_DIR, 'model_nn_metadata.json')
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, 'model_nn.onnx')

# League Draft Domain Config
ROLES = ['top', 'jungle', 'mid', 'bot', 'support']
VALID_RIOT_ROLES = {'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY'}

# Model Architecture Hyperparameters
EMBEDDING_DIM = 8
NUM_EXTRA_FEATURES = 44
