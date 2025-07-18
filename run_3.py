from training import train

if __name__ == "__main__":
        
        for seed in [1,2,3,4]:        
                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=0,
                        mapping_approach='polynomial_custom'
                        )
                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=1,
                        trivial_augment=2,
                        random_erasing=0,
                        mapping_approach='polynomial_custom'
                        )
                