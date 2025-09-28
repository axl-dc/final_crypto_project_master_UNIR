import os
ROOT_DIR   = os.path.dirname(__file__)
DB_PATH    = os.path.join(ROOT_DIR, "data", "crypto_data.sqlite")
TABLE      = "btc_1h"
INTERVAL_MS = 600
MIN_SAMPLE  = 300
N_UPDATE    = 3
LOOKBACK_FRAC = 0.6