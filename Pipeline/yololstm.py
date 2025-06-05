from ultralytics import YOLO
import torch
import torch.nn as nn

class YOLO_LSTM(nn.Module):
    """
    A combined YOLO backbone and LSTM model for sequence classification.
    The YOLO backbone is frozen to act as a fixed feature extractor.
    Dynamically determines backbone output feature shape.

    Args:
        yolo_model (ultralytics.YOLO): An initialized Ultralytics YOLO model.
                                       Its backbone will be used for feature extraction.
        input_image_size (tuple): The (H, W) size of the images that will be fed to YOLO backbone.
                                  (e.g., (640, 640)). Used to determine feature_shape dynamically.
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of recurrent layers in the LSTM.
        num_classes (int): The number of output classification categories.
    """
    def __init__(self, yolo_backbone: nn.Module, input_image_size: tuple = (640, 640),
                 hidden_size: int = 512, num_layers: int = 2, num_classes: int = 5):
        super().__init__()

        # ### FIX: Do NOT store the full ultralytics.YOLO object as a direct submodule.
        # This was the root cause of the FileNotFoundError and incorrect parameter counts.
        # Instead, directly extract the PyTorch nn.Module backbone.
        
        # Access the underlying PyTorch model and truncate it to the backbone layers (e.g., first 10 for YOLOv8n)
        # yolo_model.model.model is the nn.Module that holds the actual layers.
        self.yolo_backbone = nn.Sequential(*yolo_backbone) # For YOLOv8n, this is the backbone

        # --- FREEZE YOLO BACKBONE PARAMETERS ---
        for param in self.yolo_backbone.parameters():
            param.requires_grad = False
         # Confirm freezing
        num_trainable = sum(p.numel() for p in self.yolo_backbone.parameters() if p.requires_grad)
        print(f"YOLO backbone trainable params: {num_trainable}")  # Should be 0

        # Set the backbone to evaluation mode (important for BatchNorm/Dropout layers to behave correctly for inference)
        self.yolo_backbone.eval() 

        # --- Dynamically Determine Feature Shape from YOLO Backbone ---
        # Create a dummy input tensor matching the expected input_image_size (B=1)
        dummy_input_for_feature_shape = torch.randn(1, 3, input_image_size[0], input_image_size[1])
        with torch.no_grad():
            dummy_backbone_output = self.yolo_backbone(dummy_input_for_feature_shape)
        
        # DEBUG print to confirm shape
        print(f"DEBUG: Shape of dummy_backbone_output: {dummy_backbone_output.shape}")

        # Adapt feature shape extraction based on the actual output dimensions
        output_shape = dummy_backbone_output.shape
        if len(output_shape) == 4:
            # Expected (B, C, H, W)
            C_feat, H_feat, W_feat = output_shape[1:]
        elif len(output_shape) == 3:
            # If output is (B, C, S) or (B, S, C) - assuming (B, C, H) and W=1
            C_feat, H_feat = output_shape[1:]
            W_feat = 1 
            print(f"Warning: Backbone output is 3D {output_shape}. Assuming (C, H) and W=1 to unpack features.")
        elif len(output_shape) == 2:
            # If output is (B, Features) - already flattened
            C_feat = output_shape[1]
            H_feat = 1
            W_feat = 1
            print(f"Warning: Backbone output is 2D {output_shape} (already flattened). Assuming H=1, W=1.")
        else:
            raise ValueError(f"Unexpected backbone output shape: {output_shape}. Expected 2D, 3D, or 4D.")

        self.feature_shape = (C_feat, H_feat, W_feat)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce H, W to 1x1
        # Calculate the input dimension for the LSTM based on the dynamically determined feature_shape
        self.input_dim = C_feat
        print(f"Dynamically determined YOLO backbone feature shape: {self.feature_shape}")
        print(f"Calculated LSTM input dimension: {self.input_dim} (C*H*W)")

        # Initialize the LSTM layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        print(f"LSTM initialized with hidden_size={hidden_size}, num_layers={num_layers}")

        # Initialize the final fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        print(f"Fully connected layer initialized with {hidden_size} -> {num_classes} outputs")

    def forward(self, x_seq: torch.Tensor):
        """
        Forward pass through the YOLO backbone and LSTM.

        Args:
            x_seq (torch.Tensor): Input sequence of images.
                                  Shape: (B, T, 3, H, W)
                                  B: Batch size
                                  T: Number of timesteps (frames in sequence)
                                  3: RGB channels
                                  H, W: Image height and width (e.g., 640, 640)

        Returns:
            torch.Tensor: Logits for classification. Shape: (B, num_classes)
        """
        B, T, C, H, W = x_seq.shape
        features = []

        with torch.no_grad():
            for t in range(T):
                current_frame_batch = x_seq[:, t]
                
                f = self.yolo_backbone(current_frame_batch) # ### FIX: Removed [-1] here
                f = self.pool(f)  # (B, C, 1, 1)
                f = f.view(B, -1)  # (B, C)
                # Ensure the output is 4D before flattening if it's 3D (B, C, S)
                if len(f.shape) == 3:
                    f = f.unsqueeze(-1) # Add a W dimension of 1
                elif len(f.shape) == 2:
                    pass # f is already (B, Features)
                else:
                    pass 

                f = f.view(B, -1) # Flatten for LSTM
                features.append(f)

        lstm_input = torch.stack(features, dim=1)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        final_lstm_output = h_n[-1]
        logits = self.fc(final_lstm_output)

        return logits
