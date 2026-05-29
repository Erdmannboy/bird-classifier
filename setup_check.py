# ./setup_check.py
import sys

import matplotlib
import numpy
import pandas
import plotly
import sklearn
import streamlit


def main() -> None:
    print("ML4B Setup-Test")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"numpy: {numpy.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"plotly: {plotly.__version__}")
    print(f"streamlit: {streamlit.__version__}")
    print("Setup ist korrekt.")


if __name__ == "__main__":
    main()