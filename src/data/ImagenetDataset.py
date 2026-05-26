from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset


class ImagenetDataset(Dataset):
    def __init__(self, aumento_datos, split):
        self.aumento_datos = aumento_datos
        self.dataset = load_dataset("ILSVRC/imagenet-1k")[split]
        # DA:
        #  -> codigo: https://github.com/Ma-Lab-Berkeley/CRATE/blob/674408fa82475fe1f172aa8213e21d4ba608afc4/main.py#L254
        #  -> paper:  we only apply the standard techniques, random cropping and random horizontal flipping
        if aumento_datos:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        #    return min(1000, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx):
        pil_image, label = self.dataset[idx]["image"], self.dataset[idx]["label"]
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return self.transform(pil_image), label

    def get_num_classes(self):
        return 1000
