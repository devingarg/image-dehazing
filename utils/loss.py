from torch import nn

class HybridLoss(nn.Module):
    def __init__(self, net=None, gamma=0.0, ignore_percept=False):
        super(HybridLoss, self).__init__()
        
        # the network being used to get embeddings
        self.net = net
        
        # the weight given to the pixel_loss
        self.gamma = gamma
        
        # flag to ignore the perceptual loss and just use the pixel loss
        self.ignore_percept = ignore_percept
        
    def forward(self, output, target):
        
        # pixel wise loss
        pixel_loss = nn.MSELoss()(output, target)
        
        if self.ignore_percept:
            return pixel_loss
        
        self.net.eval()

        # Extract embeddings from the net for output and target
        output_features = self.net(output)
        target_features = self.net(target)

        
        # Compute the L2 distance between the two in the feature space
        perceptual_loss = nn.MSELoss()(output_features, target_features)
        
        return perceptual_loss + self.gamma * pixel_loss