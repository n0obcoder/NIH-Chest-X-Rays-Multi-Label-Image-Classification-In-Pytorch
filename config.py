pkl_dir_path = 'pickles'
train_val_df_pkl_path = 'train_val_df.pickle'
test_df_pkl_path = 'test_df.pickle'
disease_classes_pkl_path = 'disease_classes.pickle'
models_dir = 'models'


from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# transforms.RandomHorizontalFlip() not used because some disease might be more likely to the present in a specific lung (lelf/rigth)
transform = transforms.Compose([transforms.ToPILImage(), 
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    normalize])
