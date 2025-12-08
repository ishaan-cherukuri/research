\
import os
from pathlib import Path
from nilearn import plotting as niplt

def qc_overlay(anat_path, overlay_path, out_png, title=''):
    Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
    disp = niplt.plot_stat_map(overlay_path, bg_img=anat_path, display_mode='ortho',
                               title=title, cmap='hot', dim=0.5)
    disp.savefig(out_png, dpi=120)
    disp.close()
