import uuid

import cv2
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from torch import nn
from VGG16 import VGG16


model = VGG16()
model.load_state_dict(torch.load('fire_or_not_fire.pth'))
model.eval()


def get_fps(video):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        return int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    else:
        return int(video.get(cv2.CAP_PROP_FPS))


def evaluate_fire(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 244]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
    ])

    with torch.no_grad():
        image = transform(frame)
        image = image.unsqueeze(0)

        output = model(image.float())

        prediction = nn.functional.softmax(output, dim=None)

    return prediction[0][0].item() * 100.00


def main():
    cap = cv2.VideoCapture('test_video.mp4')
    fps = get_fps(cap)
    count = 0
    prob = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if count % fps == 0:
            prob = evaluate_fire(frame)

        cv2.putText(frame, "Fire prob: %.3f" % prob, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()