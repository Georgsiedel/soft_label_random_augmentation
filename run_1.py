from training import train

if __name__ == "__main__":
       
        for seed in [1,2]:
                
                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=1,
                        patch_gaussian=1,
                        random_erasing_p=0.25,
                        random_erasing_max_scale=0.8,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        patch_gaussian=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.2,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=1,
                        patch_gaussian=3,
                        random_erasing_p=0.25,
                        random_erasing_max_scale=0.8,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR10",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        patch_gaussian=4,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.2,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=1,
                        patch_gaussian=1,
                        random_erasing_p=0.25,
                        random_erasing_max_scale=0.8,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        patch_gaussian=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.2,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=1,
                        patch_gaussian=3,
                        random_erasing_p=0.25,
                        random_erasing_max_scale=0.8,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="CIFAR100",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        patch_gaussian=4,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.2,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=1,
                        patch_gaussian=1,
                        random_erasing_p=0.25,
                        random_erasing_max_scale=0.8,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        patch_gaussian=2,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.2,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=1,
                        patch_gaussian=3,
                        random_erasing_p=0.25,
                        random_erasing_max_scale=0.8,
                        reweight=False
                        )
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=2,
                        trivial_augment=1,
                        random_erasing=2,
                        patch_gaussian=4,
                        random_erasing_p=0.75,
                        random_erasing_max_scale=0.2,
                        reweight=False
                        )