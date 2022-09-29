def show_transformed_images(dataset):
    loader = train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True )
    batch = next(iter(loader))
    images, labels = batch
    
    grid = torchvision.utils.make_grid(images, nrow =3)
    
    plt.figure(figsize = (11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    

show_transformed_images(dataset)






FILE = "model.pth"
torch.save(the_model, FILE)