from training import train

if __name__ == "__main__":
        
        for seed in [0]:
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=2,
                        random_erasing=0,
                        reweight=True
                        )

                