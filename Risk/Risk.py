import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import gaussian_kde

df = pd.read_excel(r'C:\Users\Jhona\OneDrive - Grupo Marista\Clube de Finanças\CBC Capital\Planilhas\PRBR11 06-09.xlsx', index_col=0, parse_dates=True)

dados = df.pct_change().dropna()

## Resample os dados em semanas
weekly_maxima = dados.resample('W').max()

### Plotando dados
plt.style.use('ggplot')
fig, ax = plt.subplots(2, 1)
ax[0].plot(weekly_maxima.index, weekly_maxima['CBC Capital'], color='r')
ax[1].plot(weekly_maxima.index, weekly_maxima['PRBR11'], color='r')
ax[0].set_title('Máximo drowdown para cada semana')
plt.show()

### Parametrizando VaR e CVaR via KDE
params = t.fit(dados['CBC Capital'])
kde = gaussian_kde(dados['CBC Capital'])

sample = kde.resample(size=1000)
VaR_99 = np.quantile(sample, 0.99)
print(VaR_99)

# Find the VaR as a quantile of random samples from the distributions
VaR_99_T   = np.quantile(t.rvs(size=1000, *params), 0.99)
VaR_99_KDE = np.quantile(kde.resample(size=1000), 0.99)

# Find the expected tail losses, with lower bounds given by the VaR measures
integral_T   = t.expect(lambda x: x, args = (params[0],), loc = params[1], scale = params[2], lb = VaR_99_T)


# Create the 99% CVaR estimates
CVaR_99_T   = (1 / (1 - 0.99)) * integral_T

# Display the results
print(VaR_99_T)
print(VaR_99_KDE)
print(CVaR_99_T)