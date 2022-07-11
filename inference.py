import numpy as np
from zmq import device
import cv2
import torch

class BaseTeamPlayerSeperator:
    def __init__(self,pretrained_model,url_to_model):
        self.pretrained_model = pretrained_model
        self.url_to_model = url_to_model
        self.load_base_algorithm()

    def load_base_algorithm(self): 
        try:
            import sys
            sys.path.insert(0, './yolov5')
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=self.pretrained_model, force_reload=True) 
            # if self.pretrained_model else torch.hub.load('ultralytics/yolov5', 'yolov5s', path=self.url_to_model, force_reload=True)
            self.model = torch.load(self.url_to_model) if not self.pretrained_model else self.model
            torch.save(self.model, self.url_to_model) if self.pretrained_model else None

            # Test loaded model
            self.test_model_connection()
            print("Model has been loaded and tested")
            
        except Exception as e:
            raise Exception(f"Exception {e} occured while trying to load model")

    def detect_humans(self,img=None):
        img = ['https://ultralytics.com/images/zidane.jpg']  if type(img)==type(None) else img
        # batch of images

        # Inference
        results = self.model(img)
        results.save()
        # results = self.model(img)
        df_result = results.pandas().xyxy[0] # convert result to df
        df_result = df_result[df_result['class'] == 0]
        df_result = df_result[df_result['confidence'] >= 0.65]
        return df_result

    def seperator(num_of_seperators=2, color_of_seperators=['yw','be'],save_seperator=True,append_to_folder="player"):
        pass

    def save_seperators():
        pass
    
    def detect_left_top_width_height(self,x, y, w, h,image_width, image_height, width_factor=640,height_factor=640):
        x_factor = image_width / width_factor 
        y_factor =  image_height / height_factor
        left = int((x - 0.5 * w) * x_factor)
        top = int((y - 0.5 * h) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)

        return left,top, width,height

    def test_model_connection(self):
        imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

        # Inference
        results = self.model(imgs)

        # Results
        results.print()
        results.save()  # or .show()

        results.xyxy[0]  # img1 predictions (tensor)
        print(results.pandas().xyxy[0])

class ImagePlayerSeperator(BaseTeamPlayerSeperator):
    def __init__(self,pretrained_model,url_to_model):
        super(BaseTeamPlayerSeperator,self).__init__( pretrained_model=pretrained_model,url_to_model=url_to_model)

    def image_inference(self,image, is_url=False):
        # Read the image
        image = cv2.imread(image) if is_url else image #expecting a numpy array or url to the image
        image_width, image_height = image.shape[0],image.shape[1]
        df_result = self.detect_humans(image)
        x, y, w, h = df_result.iloc[0]['xmin'], df_result.iloc[0]['ymin'], df_result.iloc[0]['xmax'], df_result.iloc[0]['ymax']
        left,top, width,height = self.detect_left_top_width_height(x, y, w, h,image_width, image_height)
        print(f"image shape {image.shape}")
        print(f"x, y, w, h: {x},{y}, {w},{h}")
        print(f"left,top, width,height: {left},{top}, {width},{height}")
        # cv2.imshow('Frame',image[y:h,x:w])
        print(f"image shap")
        roi_color = image[top:top+height, left:left+width]
        cv2.imshow('Frame',roi_color)
        df_seperators = self.seperator(df_result)
        return df_result, df_seperators


class VideoPlayerSeperator(ImagePlayerSeperator):
    def __init__(self,pretrained_model,url_to_model):
        super(ImagePlayerSeperator,self).__init__( pretrained_model=pretrained_model,url_to_model=url_to_model )

    def video_inference(self,video_file):
        cap = cv2.VideoCapture(video_file)
        ret, frame = cap.read()

        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        # Read until video is completed
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                # cv2.imshow('Frame',frame)

                # Detect positions of human in image if any exist
                df_result, df_seperators = self.image_inference(frame)
                print(f"df_result, df_seperators, {df_result}, {df_seperators}")
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()