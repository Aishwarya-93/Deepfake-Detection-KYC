from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading real and fake face images.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = []
        self.labels = []

        classes = {
            "real": 0,
            "fake": 1
        }

        for class_name, label in classes.items():

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for image in class_dir.glob("*"):

                if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.image_paths.append(image)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])