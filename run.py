from training import train

if __name__ == "__main__":
        
        
        for seed in [0]: 
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=True
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=2,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=2,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=True
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=True
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=True
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=0,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=True
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=2,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=2,
                        trivial_augment=0,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=True
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4,
                        reweight=True
                        )
        for seed in [1,2]:    
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=0,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.4
                        ) 
                