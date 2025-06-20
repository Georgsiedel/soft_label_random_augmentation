from training import train

if __name__ == "__main__":
        
        for seed in [4]:        

                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=0,
                        patch_gaussian=4
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=0,
                        patch_gaussian=2
                        )