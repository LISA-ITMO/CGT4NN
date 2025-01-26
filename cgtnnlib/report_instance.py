import os


from cgtnnlib.Report import Report
from cgtnnlib.constants import REPORT_DIR


if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

report = Report(dir=REPORT_DIR)