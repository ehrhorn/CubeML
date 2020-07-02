from src.modules.db_utils import *
from src.modules.classes import SqliteFetcher
from src.modules.constants import *
import joblib

old_db = SqliteFetcher(PATH_DATA_OSCNEXT+'/val_set_sqlite.db')
new_db = SqliteFetcher(PATH_DATA_OSCNEXT+'/val_transformed.db')
transformers = joblib.load(open(PATH_TRANSFORMERS, 'rb'))

create_transformed_db(old_db, new_db, transformers)