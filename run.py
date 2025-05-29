from training import train

if __name__ == "__main__":
        
        for seed in [0]: 
                
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=2,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False,
                        mapping_approach="other")
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=2,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=True,
                        mapping_approach="other")
                
        for seed in [3]:        
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=True)
                
        for seed in [1,2,3]:
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.6,
                        reweight=True)
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.8,
                        reweight=True)
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=1.0,
                        reweight=True)
        
        for seed in [0]: 
                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False)
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False)
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=0,
                        random_erasing_p=0.5,
                        random_erasing_max_scale=0.4,
                        reweight=False)
                