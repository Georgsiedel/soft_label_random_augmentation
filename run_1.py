from training import train

if __name__ == "__main__":

                
        for seed in [0,1,2,3,4]:        
                train(seed = seed,
                        dataset="TinyImageNet",
                        random_cropping=1,
                        trivial_augment=1,
                        random_erasing=0,
                        mapping_approach='polynomial_custom'
                        )
                