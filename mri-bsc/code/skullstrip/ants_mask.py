\
import argparse, ants

def run_ants_mask(t1_in, mask_out):
    img = ants.image_read(t1_in)
    m = ants.get_mask(img)
    ants.image_write(m, mask_out)
    print(f"Saved mask -> {mask_out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--t1', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    run_ants_mask(args.t1, args.out)
