if __name__ == "__main__":
    import torch
    from tqdm import tqdm 
    import torchvision.transforms as transforms
    from torchvision import models
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torch import nn, optim
    from PIL import ImageFile, Image
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # <- this fixes truncated images


    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    image_size = 299
    epochs = 10
    model_save_path = './fire_detection_model.pth'

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Datasets and loaders
    train_data = ImageFolder('./dataset/train', transform=transform)
    val_data = ImageFolder('./dataset/val', transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model (with updated weights handling)
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Add tqdm progress bar for training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}] Training')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # handle aux_logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Update progress bar description with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

        model.eval()
        correct = total = 0
        # Add tqdm progress bar for validation loop
        val_pbar = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{epochs}] Validation')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = torch.sigmoid(model(inputs))
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predicted = (outputs > 0.5).squeeze().long()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar description with current accuracy
                val_pbar.set_postfix({'acc': f'{correct/total:.4f}'})

        print(f"Validation Accuracy: {correct / total:.4f}")

    torch.save(model.state_dict(), model_save_path)
