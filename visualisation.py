import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from fit_psyche.psychometric_curve import PsychometricCurve


def print_results(all_blocks, print=None):
    
    box_style = {
        'whiskerprops': {'linewidth': 0.5, 'color': 'gray'},
        'medianprops': {'linewidth': 0.5, 'color': 'gray'},
        'boxprops': {'color': 'gray', 'linewidth': 0.5},
        'capprops': {'color': 'gray', 'linewidth': 0.5}
    }

    colours = ["orange", "magenta"]
    x = np.linspace(-0.2, 1.2, num=500)
    f, ax = plt.subplots(1,2, figsize=(10, 5))
    for ix, cond in enumerate(all_blocks.counterfactual_label.unique()):
        
        ax[ix].set_ylim(-0.1, 1.1)
        ax[ix].set_ylabel("proportion of correct responses")
        ax[ix].set_xlim(-0.1, 0.6)
        ax[ix].set_xlabel("coherence [proportion of target dots]")
        ax[ix].set_xticks(np.linspace(0.0, 0.5, num=6))
        coh = []
        prob = []
        cond_data = all_blocks.loc[
            (all_blocks.counterfactual_label == cond)
        ]
        coherence, prob_correct = np.split(cond_data.groupby("coherence_level").response_correct.mean().reset_index().to_numpy(), 2, 1)
        coherence = coherence.flatten()
        prob_correct = prob_correct.flatten()
        ax[ix].scatter(coherence, prob_correct, s=50, color=colours[ix], label="Data", lw=1, edgecolors="black")
        wh = PsychometricCurve(model='wh', guess_rate_lims=[0.0, 0.000001]).fit(coherence, prob_correct)
        curve = wh.predict(x)
        ax[ix].plot(x, curve, label=f"Curve fit", c=colours[ix])
        ax[ix].set_title(cond)
        ax2 = ax[ix].twinx()
        ax2.set_ylim(-10, 110)
        ax2.set_ylabel("confidence [%]")

        coh, mean_conf = np.split(cond_data.loc[cond_data.response_correct==True].groupby(["coherence_level"]).scale_response.mean().reset_index().to_numpy(),2, 1)
        coh = coh.flatten()
        mean_conf = mean_conf.flatten() * 100
        ax2.plot(coh - 0.01, mean_conf, lw=2, alpha=0.5, c="green")

        coh, mean_conf = np.split(cond_data.loc[cond_data.response_correct==False].groupby(["coherence_level"]).scale_response.mean().reset_index().to_numpy(),2, 1)
        coh = coh.flatten()
        mean_conf = mean_conf.flatten() * 100
        ax2.plot(coh + 0.01, mean_conf, lw=2, alpha=0.5, c="red")

        for coh in cond_data.coherence_level.unique():
            resp = cond_data.loc[
                (cond_data.response_correct==True) &
                (cond_data.coherence_level== coh)
            ].scale_response.to_numpy() * 100

            bplot = ax2.boxplot(resp, positions=[coh-0.01], widths=0.01, showfliers=False, manage_ticks=False, **box_style)
            bplot["boxes"][0].set_color("green")
        for coh in cond_data.coherence_level.unique():
            resp = cond_data.loc[
                (cond_data.response_correct==False) &
                (cond_data.coherence_level== coh)
            ].scale_response.to_numpy() * 100

            bplot = ax2.boxplot(resp, positions=[coh+0.01], widths=0.01, showfliers=False, manage_ticks=False, **box_style)
            bplot["boxes"][0].set_color("red")
        ax[ix].legend(loc=4, fontsize="xx-small")
        custom_lines = [
            Line2D([0], [0], color="green", lw=2, alpha=0.5, label="Mean confidence correct"),
            Line2D([0], [0], color="red", lw=2, alpha=0.5, label="Mean confidence incorrect"),
        ]

        ax2.legend(handles=custom_lines, loc=8, fontsize="xx-small")

    f.suptitle(f"subject: {all_blocks.subject.unique().tolist()[0]}, 0.03 low level reversed")
    plt.tight_layout()
    if print != None:
        f.savefig(print)