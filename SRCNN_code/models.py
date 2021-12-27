from torch import nn

# MODIFIED according to the results presented by the FSRCNN paper and the EDSR paper
# PReLU instead of ReLU for nonlinearity 
# Filter size decreased to 9-5-9
# Feedback of "x" to transition to a residual block 

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        # self.relu = nn.ReLU(inplace=True)
        # self.sig = nn.Sigmoid()
        self.prelu = nn.PReLU()
        # self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        identity = x
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        # x = self.lrelu(self.conv1(x))
        # x = self.lrelu(self.conv2(x))
        # x = self.sig(self.conv1(x))
        # x = self.sig(self.conv2(x))
        x = self.conv3(x)
        x += identity
        return x
