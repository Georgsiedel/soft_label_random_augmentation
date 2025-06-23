from training import train

if __name__ == "__main__":

        for seed in [4]:

                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=0,
                        patch_gaussian=1
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=0,
                        patch_gaussian=2
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=0,
                        patch_gaussian=3
                        )
                