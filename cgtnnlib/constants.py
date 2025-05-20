## Constants v.0.2
## Created at Tue 14 Jan 2025
## Updated at Sun 19 Jan 2025

ITERATIONS = 10
REPORT_DIR = "report/"
MODEL_DIR = "pth/"
DATA_DIR = "data"
DRY_RUN = False
EPOCHS = 20
LEARNING_RATE = 0.00011
TEST_SAMPLE_SIZE = 0.2
RANDOM_STATE = 23432
BATCH_SIZE = 12
PMLB_TARGET_COL = "target"
NOISE_SAMPLES_COUNT = 50
NOISE_FACTORS = [x * 2 / NOISE_SAMPLES_COUNT for x in range(NOISE_SAMPLES_COUNT)]

# In [1]: import cgtnnlib.constants as c#
# In [4]: len(c.NOISE_FACTORS)
# Out[4]: 50
# In [5]: c.NOISE_FACTORS[0]
# Out[5]: 0.0
# In [6]: c.NOISE_FACTORS[49]
# Out[6]: 1.96
# In [7]: def n(i):
#    ...:     return c.NOISE_FACTORS[i]
#    ...:
# In [8]: n(0)
# Out[8]: 0.0
# In [9]: n(49)
# Out[9]: 1.96
# In [10]: [x for x in [n(0), n(9), n(19), n(29), n(39), n(49)]]
# Out[10]: [0.0, 0.36, 0.76, 1.16, 1.56, 1.96]
