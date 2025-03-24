import cv2

def videofeed():
    
    cam = cv2.VideoCapture(0)
    
    while True:
        
        ret, frame = cam.read()
        
        if cv2.waitKey(1) == ord('q'):
            break
    
        cv2.line(frame, (300, 203), (400, 203), (0,123,153), 5)
    
        cv2.imshow('feed', frame)
        
        
        
    cv2.destroyAllWindows()
    cam.release()
    
    
if __name__ == "__main__":
    
    videofeed()
    