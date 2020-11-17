from models.wgan_gradient_penalty import WGAN_GP
import argparse
import torchvision,torch
import torchvision.transforms as transforms

def main(args):
    #Create WGAN model
    model = WGAN_GP(args)

    #Convert to grayscale and resize transform
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop((32, 32), padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,)),
    ])
    data_path = 'UCSD/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=trans
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )

    # Start model training
    model.train(train_loader)

if __name__ == '__main__':
    args = argparse.Namespace()
    args.generator_iters = 40000
    args.cuda = "True"
    args.batch_size = 64
    args.channels = 1
    args.load_weight=False
    args.plot=False # do you wnat to plot loss function or not?
    main(args)


