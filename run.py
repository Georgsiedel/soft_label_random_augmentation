from training import train

if __name__ == "__main__":

        runs = 1
        for seed in range(runs):

                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=0,
                        random_erasing=0,
                        random_erasing_p=0.3,
                        random_erasing_max_scale=0.33,
                        reweight=False,
                        mapping_approach="fixed_params")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=0,
                        random_erasing_p=0.3,
                        random_erasing_max_scale=0.33,
                        reweight=False,
                        mapping_approach="fixed_params")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=0,
                        random_erasing_p=0.3,
                        random_erasing_max_scale=0.33,
                        reweight=False,
                        mapping_approach="fixed_params")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=1,
                        random_erasing_p=0.3,
                        random_erasing_max_scale=0.33,
                        reweight=False,
                        mapping_approach="fixed_params")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=0,
                        random_erasing=1,
                        random_erasing_p=0.3,
                        random_erasing_max_scale=0.33,
                        reweight=False,
                        mapping_approach="fixed_params")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=1,
                        random_erasing_p=0.3,
                        random_erasing_max_scale=0.33,
                        reweight=False,
                        mapping_approach="fixed_params")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=1,
                        random_erasing_p=0.3,
                        random_erasing_max_scale=0.33,
                        reweight=False,
                        mapping_approach="fixed_params")
                