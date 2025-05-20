from training import train

if __name__ == "__main__":

        # Example configuration; tweak as needed:
        config = dict(
                dataset="CIFAR10",
                random_cropping=1,
                trivial_augment=0,
                random_erasing=0,
                random_erasing_p=0.3,
                random_erasing_max_scale=0.33,
                reweight=False,
                mapping_approach="fixed_params",
        )

        runs = 1

        # If you want multiple runs with different seeds:
        for seed in range(runs):
                config["seed"] = seed
                train(**config)
