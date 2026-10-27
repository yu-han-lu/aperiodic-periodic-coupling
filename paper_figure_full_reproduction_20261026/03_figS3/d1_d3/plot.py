from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common import plot_2d
plot_2d("linear", "d1_d3", HERE)
