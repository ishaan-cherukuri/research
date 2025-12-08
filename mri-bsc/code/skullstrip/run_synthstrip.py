\
import argparse, os, subprocess, shutil

def run_synthstrip(t1_in, brain_out, mask_out):
    exe = shutil.which('mri_synthstrip')
    if exe is None:
        raise RuntimeError('mri_synthstrip not found in PATH. Install FreeSurfer and ensure it is available.')
    cmd = [exe, '-i', t1_in, '-o', brain_out, '-m', mask_out]
    subprocess.run(cmd, check=True)
    print(f"SynthStrip OK -> brain:{brain_out} mask:{mask_out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--t1', required=True)
    ap.add_argument('--brain', required=True)
    ap.add_argument('--mask', required=True)
    args = ap.parse_args()
    run_synthstrip(args.t1, args.brain, args.mask)
