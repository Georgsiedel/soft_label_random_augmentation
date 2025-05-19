import os
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':

    runs = 1
    run_iter =[0]

    for run in run_iter:

        print("Training run #",run)
        cmd0 = f"python train.py --seed={run} --dataset=CIFAR10 --random_cropping=1 --trivial_augment=0 " \
                f"--random_erasing=0 --random_erasing_p=0.3 --reweight=True --random_erasing_max_scale=0.33"
        print(cmd0)
        os.system(cmd0)
