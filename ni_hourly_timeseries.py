import matplotlib.pyplot as plt

from training_datasets import get_ni_timeseries

plt. title('NI Hourly')

plt.plot(get_ni_timeseries()[:500])

plt.show()