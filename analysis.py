import pandas as pd
import os
from pathlib import Path

dead_voted = pd.read_json(Path(os.getcwd())/'out/dead_voters_who_voted.json', orient=str)
dead_registered = pd.read_json(Path(os.getcwd())/'out/registered_dead_voters.json', orient=str)

registered = pd.read_csv(Path(os.getcwd())/'mi_wa_voterfile.csv')
