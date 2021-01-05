## Coby Penso 208254128 ##

import argparse
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
import numpy as np
from VAE import Model

# Support running on GPU and CPU #
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def train(vae, trainloader, optimizer, epoch):
    '''
        Train - Single epoch train
        @params:    vae - the model to be trained and optimized
                    trainloader - loader of the dataset to train on
                    optimizer
                    epoch - the current epoch
    '''

    vae.train()  # set to training mode
    
    loss_trace = []
    
    for data in trainloader:
        x, _ = data
        x = x.to(device)
        
        optimizer.zero_grad()
        loss = vae(x)
        loss.backward()
        optimizer.step()
        loss_trace.append(-loss.item()/len(x)) # noramlize the loss by the batch size

    epoch_loss = np.mean(loss_trace)
    print ("Epoch: {}, Loss: {}".format(epoch, epoch_loss))

    return epoch_loss


def test(vae, testloader, filename, epoch, total_epochs, sample_size):
    '''
        Test - test the model
        @params:    vae - the model to test
                    testloader - loader of the dataset to test on
                    filename - path to save the sampled data from the model
                    epoch - the current epoch
                    total_epochs - number of total epochs
                    sample_size - number of sample to generate from the model
    '''

    vae.eval()  # set to inference mode
    
    with torch.no_grad():
        if (epoch % 10 == 0) or (epoch == (total_epochs - 1)):
            samples = vae.sample(sample_size).cpu()
            samples.clamp_(0,1)
            torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                         './samples/' + filename + 'epoch%d.png' % (epoch+1))
        
        loss_trace = []
        for data in testloader:
            x, _ = data
            x =  x.to(device)
                
            loss = vae(x)
            loss_trace.append(-loss.item()/len(x))  # noramlize the loss by the batch size
        
        test_loss = np.mean(loss_trace)
        print ("Epoch: {}, Test Loss: {}".format(epoch, test_loss))
        return test_loss

def visualize_elbo(train_loss, test_loss):
    '''
        Visualize the loss trace through the training process

        @note: show on the same figure the train loss and test loss
    '''
    plt.figure()       
    # Plot both train and test loss on the same figure
    plt.plot(train_loss, linewidth=3, color='red', label='Train ELBO')
    plt.plot(test_loss, linewidth=3, color='green', label='Test ELBO')
    plt.legend()
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('ELBO', fontsize=12)
    plt.grid(True)
    # Save the plot as a png image
    plt.savefig('train&test_ELBO.png')

def main(args):
    
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)), #dequantization
        transforms.Normalize((0.,), (257./256.,)), #rescales to [0,1]

    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    vae = Model(latent_dim=args.latent_dim,device=device).to(device)
    optimizer = torch.optim.Adam(
        vae.parameters(), lr=args.lr)
    
    train_loss_trace = []
    test_loss_trace= []
    for epoch in range(args.epochs):
        # Train Phase
        loss = train(vae, trainloader, optimizer, epoch)   
        train_loss_trace.append(loss)
        # Test Phase
        loss = test(vae, testloader, filename, epoch, args.epochs, args.sample_size)
        test_loss_trace.append(loss)
        
    # Plot the train and test loss trace
    visualize_elbo(train_loss_trace, test_loss_trace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=10)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
