from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    image_path = f'../../train/0a0eaeaf-9ad0c6dd/0a0eaeaf-9ad0c6dd-8.png'

    image = Image.open(image_path)

    affine_transform = transforms.RandomAffine(degrees=15, translate=(0,0.2))
    perspective_transform = transforms.RandomPerspective(p=1, distortion_scale=0.2)

    transformed_image = perspective_transform(image)

    transformed_image.show()

    image.save('images/aug.png')
    transformed_image.save('images/aug-transformed-pers.png')