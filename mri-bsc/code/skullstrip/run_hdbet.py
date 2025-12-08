\
import argparse, os, subprocess, shutil, tempfile

def run_hdbet(t1_in, mask_out, device='cpu'):
    exe = shutil.which('hd-bet')
    if exe is None:
        raise RuntimeError('hd-bet CLI not found. `pip install hd-bet` (requires torch)')
    out_base = tempfile.mktemp()
    cmd = [exe, '-i', t1_in, '-o', out_base, '-device', device]
    subprocess.run(cmd, check=True)
    msk = f"{out_base}_mask.nii.gz"
    if not os.path.exists(msk):
        raise RuntimeError('HD-BET mask not found')
    os.replace(msk, mask_out)
    print(f"HD-BET OK -> mask:{mask_out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--t1', required=True)
    ap.add_argument('--mask', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    run_hdbet(args.t1, args.mask, device=args.device)
