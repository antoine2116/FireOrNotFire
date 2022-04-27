import uuid

import cv2
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from torch import nn
from VGG16 import VGG16
from model_prof import FireNet

model = FireNet()
model.load_state_dict(torch.load('fire_or_not_fire_b100_lr0005.pth'))
model.eval()


def get_fps(video):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        return int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    else:
        return int(video.get(cv2.CAP_PROP_FPS))


def evaluate_fire(image):
    prediction_transform = transforms.Compose([transforms.Resize(size=(64, 64)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3, :, :].unsqueeze(0)
    output = model(image)
    pred = nn.functional.softmax(output)
    return pred[0][1].item() * 100


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
            image = frame.copy()
            image = transforms.ToPILImage()(image)
            prob = evaluate_fire(image)

        cv2.putText(frame, "Fire prob: %.3f" % prob, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()