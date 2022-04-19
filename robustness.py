import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lqg import xcorr
from lqg.tracking import SubjectiveVelocityModel

if __name__ == '__main__':
    data = []
    for sigma in [1., 10.]:
        for subj_vel_noise in [0., 1., 10.]:
            model = SubjectiveVelocityModel(process_noise=1., subj_noise=1.,
                                            c=.5, motor_noise=.5,
                                            subj_vel_noise=subj_vel_noise, sigma=sigma)

            x = model.simulate(n=20, T=1000)

            lag, ccg = xcorr(np.diff(x[..., 1], axis=0).T, np.diff(x[..., 0], axis=0).T, maxlags=60)
            plt.plot(lag, ccg.mean(axis=0))

            data.append(dict(subj_vel_noise=subj_vel_noise, sigma=sigma, mse=np.mean((x[..., 1] - x[..., 0]) ** 2)))

    plt.show()

    df = pd.DataFrame(data)
