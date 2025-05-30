from training import train

if __name__ == "__main__":

        for seed in [1,2,3,4]:

                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=2,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False,
                        mapping_approach="exact_model_accuracy")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=2,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False,
                        mapping_approach="polynomial_chance")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=2,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False,
                        mapping_approach="polynomial_custom")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=2,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False,
                        mapping_approach="exact_hvs")
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=2,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False,
                        mapping_approach="smoothened_hvs_or_model_accuracy")