from training import train

if __name__ == "__main__":

        for seed in [3,4]:

                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=1.0,
                        random_erasing_max_scale=0.8,
                        reweight=False,
                        mapping_approach="fixed_params")
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=1.0,
                        random_erasing_max_scale=1.0,
                        reweight=False,
                        mapping_approach="fixed_params")