from evidently.report import Report
from evidently.metrics import DataDriftTable
import pandas as pd

reference_data = pd.read_csv('data/processed/train.csv')
current_data = pd.read_csv('data/processed/current.csv')

report.run(reference_data=reference_data, current_data=current_data)
report.save_html('monitoring/reports/drift_report.html')