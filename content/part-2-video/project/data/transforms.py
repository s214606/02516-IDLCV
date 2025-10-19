from torchvision import transforms as T

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor()
    ])