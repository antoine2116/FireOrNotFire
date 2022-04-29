import cv2
import torch
import torchvision.transforms as transforms

from firenet import FireNet

# Load model
model = FireNet()
model.load_state_dict(torch.load('fire_or_not_fire_test.pth'))
model.eval()


def evaluate_fire(image):
    '''
    @input : PIL Imag
    @returns : fire prob in percentage
    '''

    # Transform function
    transform = transforms.Compose([transforms.Resize(size=(64, 64)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])

    # Transforms the image
    image = transform(image)[:3, :, :].unsqueeze(0)

    # Pass image through the CNN
    output = model(image)

    # Get prediction
    pred = torch.nn.functional.softmax(output)

    # Converts the tensor value into percentage
    return pred[0][1].item() * 100


def main():
    # Load video
    cap = cv2.VideoCapture('test_video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()

        # Breaks if video ends
        if not ret:
            break

        # Checks for kesspress event to close the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Copy frame into variable
        image = frame.copy()

        # Convert frame inrto PIL Image
        image = transforms.ToPILImage()(image)

        # Get fire probability
        prob = evaluate_fire(image)

        # Display fire probability on the frame
        cv2.putText(frame, "Fire prob: %.3f" % prob, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('frame', frame)

    # Release
    cap.release()

    # Close video window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()