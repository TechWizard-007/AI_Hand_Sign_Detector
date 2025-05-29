import cv2 as cv

var = cv.VideoCapture(0)

while True:
    istrue, frame = var.read()

    # Display the frame
    cv.imshow("Webcam", frame)

    # Wait for a key press (1 millisecond means wait for 1 ms)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

var.release()
cv.destroyAllWindows()
