DEFAULT_USER_COL = "col_user"
DEFAULT_ITEM_COL = "col_item"
DEFAULT_RATING_COL = "col_rating"
DEFAULT_LABEL_COL = "col_label"
DEFAULT_ORDER_COL = "col_order"
DEFAULT_FLAG_COL = "col_flag"
DEFAULT_TIMESTAMP_COL = "col_timestamp"
DEFAULT_PREDICTION_COL = "col_prediction"

DEFAULT_K = 10
DEFAULT_THRESHOLD = 10
MAX_N_UPDATE = 5  # ealy stop criterion, max number of epoches having no update

# implicit datasets (score being 1)
IMPLIICIT_DATASETS = [
    "ali_mobile",
    "citeulike-a",
    "citeulike-t",
    "diginetica",
    "dunnhumby",
    "gowalla",
    "delicious-2k",
    "lastfm-2k",
    "retailrocket",
    "tafeng",
    "taobao",
    "yoochoose",
]
