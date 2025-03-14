import math
import torch
import torch.nn as nn

class NeRF_model(nn.Module):
    def __init__(self, position_encoding_length=10, direction_encoding_length=6, hidden_dim=256):
        super(NeRF_model, self).__init__()
        self.position_encoding_length = position_encoding_length
        self.direction_encoding_length = direction_encoding_length
        position_size = 3 # x, y and z
        direction_size = 3 # theta and phi, but represented along X, y, z as a vector.
        embedding_size = 2 # Sin(p) and Cos(p) where p is each coordinate

        position_encoding_size = position_size * embedding_size * self.position_encoding_length
        direction_encoding_size = direction_size * embedding_size * self.direction_encoding_length

        self.activation_layer = nn.ReLU(inplace=True)
        self.sigmoid_layer = nn.Sigmoid()
        self.block1 = [
            nn.Linear(in_features=position_encoding_size + position_size, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        ]
        self.block2 = [
            nn.Linear(in_features=hidden_dim + position_encoding_size + position_size,out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        ]
        self.no_activation_layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        
        self.density_layer = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=1),
            self.activation_layer
        )
        self.color_layer = nn.Sequential(
            nn.Linear(in_features=hidden_dim + direction_encoding_size + direction_size,out_features=hidden_dim//2),
            self.activation_layer,
            nn.Linear(in_features=hidden_dim//2, out_features=3),
            self.sigmoid_layer
        )

        # Initialise the Linear layer weights
        self.initialise_weights()

    @staticmethod
    def encoding_layer(values, encoding_length):
        # param: encoding_length (L in paper)
        output = values
        for i in range(encoding_length):
            output = torch.cat((output, torch.sin((2**i)*values), torch.cos((2**i)*values)), dim=1)
        
        return output

    def forward(self, positions, directions):
        # Encode the positions and directions i.e. Gamma(x) and Gamma(d) from the paper.
        positional_encoding = self.encoding_layer(positions, self.position_encoding_length)
        directional_encoding = self.encoding_layer(directions, self.direction_encoding_length)

        intermediate_output = positional_encoding
        for layer in self.block1:
            intermediate_output = self.activation_layer(layer(intermediate_output))
        
        intermediate_output = torch.cat((intermediate_output, positional_encoding), dim=1)
        for layer in self.block2:
            intermediate_output = self.activation_layer(layer(intermediate_output))
        
        intermediate_output = self.no_activation_layer(intermediate_output)

        density = self.density_layer(intermediate_output)

        colors = self.color_layer(torch.cat((intermediate_output, directional_encoding), dim=1))

        return colors, density

    def initialise_weights(self):
        # TODO: Check if Sequential layers can be initialised through isinstance of nn.Linear.
        for module in self.block1+self.block2+[self.color_layer]+[self.no_activation_layer]+[self.density_layer]:
            if isinstance(module, nn.Linear):
                init_range = 1.0 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -init_range, init_range)
                # nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

if __name__ == "__main__":
    sample_positional_input = torch.randn(1024, 3)
    sample_directional_input = torch.randn(1024, 3)

    model = NeRF_model()

    colors, density = model(sample_positional_input, sample_directional_input)

    print("Dimension of Colors output: ", colors.shape)
    print("Dimension of Density output: ", density.shape)
