""""""
"""
   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


"""
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, StudentT
import numpy as np
#%%
x = np.arange(-10, 10, 0.01)
pdf_normal = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
pdf_studentt3 = 2 / (np.pi * np.sqrt(3) * (1 + x**2 / 3)**2)
#%%
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, pdf_normal, label='Gaussian')
ax.plot(x, pdf_studentt3, label="Student's t(3)")
ax.set_title("Gaussian vs Student's t(3)-distribution")
ax.set_xlabel('y')
ax.set_ylabel('f(y)')
ax.legend(loc='upper right')
