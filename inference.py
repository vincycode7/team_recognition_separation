import numpy as np
import cv2
import torch
import webcolors
from collections import Counter
import json

global color_seperator_mapper
# color_seperator_mapper = json.loads("color_seperator_mapper.json")
with open("color_seperator_mapper.json", 'r') as openfile:
    # Reading from json file
    color_seperator_mapper = json.load(openfile)
class BaseTeamPlayerSeperator:
    def __init__(self,pretrained_model,url_to_model,display_result_to_gui):
        self.pretrained_model = pretrained_model
        self.url_to_model = url_to_model
        self.display_result_to_gui = display_result_to_gui
        self.load_base_algorithm()
        self.check_folder_if_exist("./output") # Check the output folder if not_exist

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
        # results.save()
        # results = self.model(img)
        df_result = results.pandas().xyxy[0] # convert result to df
        df_result = df_result[df_result['class'] == 0]
        df_result = df_result[df_result['confidence'] >= 0.65]
        return df_result

    def create_bar_with_dominat_colors(self,h, w, c):
        bar = np.zeros((h,w,3), np.uint8)
        bar[:] = c
        red, green, blue = int(c[2]), int(c[1]), int(c[0])
        return bar, (red, green, blue)

    @classmethod
    def rgb2hex(cls,c):
        return "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))

    @classmethod
    def closest_colour(cls,requested_colour):
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name
        return min_colours[min(min_colours.keys())]

    @classmethod
    def closest_colour_mapper(cls,rgd_to_map):
        from sklearn.metrics import mean_squared_error
        rgd_to_map = [int(v) for v in rgd_to_map] # First round off predicted color
        h_color = cls.rgb2hex(rgd_to_map)
        try:
            nm = webcolors.hex_to_name(h_color, spec='css3')
        except ValueError as v_error:
            print("{}".format(v_error))
            rms_lst = []
            for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
                cur_clr = webcolors.hex_to_rgb(img_hex)
                rmse = np.sqrt(mean_squared_error(rgd_to_map, cur_clr))
                rms_lst.append(rmse)

            closest_color = rms_lst.index(min(rms_lst))

            nm = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]
        return nm

    def extract_dominant_color(self,roi_color,num_of_clusters):
        h,w,_ = np.shape(roi_color)
        data = np.reshape(roi_color,(h*w,3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(data,num_of_clusters,None,criteria,10,flags)
        # print(f"centers: {centers}, centers shape: {centers.shape}")

        bars = []
        rgb_values = []
        closest_colors = []

        for index, row in enumerate(centers):
            bar, rgb = self.create_bar_with_dominat_colors(200,200,row)
            colorname = self.closest_colour_mapper(tuple(row))
            # colorname = self.closest_colour(tuple(row))
            closest_colors.append(colorname)
            print(f"colorname: {colorname}")
            bars.append(bar)
            rgb_values.append(rgb)

        img_bar = np.hstack(bars)
        for index, col_name in enumerate(closest_colors):
            image = cv2.putText(img_bar, f'{index+1}, col_name: {col_name}',(5+200*index, 200-10),cv2.FONT_ITALIC,0.5,(255,0,0),1,cv2.LINE_AA)

        
        if self.display_result_to_gui:
            cv2.imshow('Dominant colours', img_bar)

        return closest_colors, img_bar

    def seperator(self,image,df_result,num_of_seperators=2, color_of_seperators=['yw','be'],save_seperator=True,append_to_folder="player"):
        for each_res in range(df_result.shape[0]):
            x, y, w, h = int(df_result.iloc[each_res]['xmin']), int(df_result.iloc[each_res]['ymin']), int(df_result.iloc[each_res]['xmax']), int(df_result.iloc[each_res]['ymax'])
            roi_color = image[y:h, x:w]
            if self.display_result_to_gui:
                cv2.imshow('Detection Frame',roi_color)
            closest_colors, color_bars_and_names = self.extract_dominant_color(roi_color=roi_color,num_of_clusters=4)
            for each_seperator in color_of_seperators:
                found_match = False
                for each_color_res in closest_colors:
                    if each_color_res in color_seperator_mapper[each_seperator]['matching']:
                        found_match = True
                        file_name = self.generate_file_name()
                        self.save_seperators("roi_image"+file_name,'./output/team_'+color_seperator_mapper[each_seperator]['name'],roi_color)
                        self.save_seperators("color_bar"+file_name,'./output/team_'+color_seperator_mapper[each_seperator]['name'],color_bars_and_names)

            # This code section will capture roi_image that were not recognised by the color recognition system
            # Uncomment to reassigned the missed samples into yellow or blue class
            # if not found_match:
            #     file_name = self.generate_file_name()
            #     self.save_seperators("roi_image"+file_name,'./output/team_'+'others',roi_color)
            #     self.save_seperators("color_bar"+file_name,'./output/team_'+'others',color_bars_and_names)

    @staticmethod
    def check_folder_if_exist(folder_path, check_if_false=True):
        # If folder doesn't exist, then create it.
        import os
        # You should change 'test' to your preferred folder.
        CHECK_FOLDER = os.path.isdir(folder_path)

        if not CHECK_FOLDER:
            os.makedirs(folder_path) if check_if_false else None
            print("created folder : ", folder_path)
        else:
            print(folder_path, "folder already exists.")
        return CHECK_FOLDER

    @classmethod
    def save_seperators(cls,file_name,folder_path, roi_color):
        # Check if path exist else create path
        '''Check if directory exists, if not, create it'''
        
        file_path = folder_path+'/'+file_name+".jpg"
        cls.check_folder_if_exist(folder_path)
        try:
            cv2.imwrite(file_path,roi_color)
        except Exception as e:
            raise Exception(f"Error {e} when trying to save image result.")
        return
    
    @staticmethod
    def generate_file_name():
        import string, random
        letters = []
        # printing lowercase
        letters += string.ascii_lowercase
        # printing uppercase
        letters += string.ascii_uppercase
        # printing letters
        letters += string.ascii_letters
        # printing digits
        letters += string.digits
        # printing punctuation
        # letters += string.punctuation
        file_name = ''.join(random.choice(letters) for i in range(20))
        return file_name

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
    def __init__(self,pretrained_model,url_to_model,display_result_to_gui):
        super(BaseTeamPlayerSeperator,self).__init__( pretrained_model=pretrained_model,url_to_model=url_to_model,display_result_to_gui=display_result_to_gui)

    def image_inference(self,image, is_url=False):
        # Read the image
        image = cv2.imread(image) if is_url else image #expecting a numpy array or url to the image
        df_result = self.detect_humans(image)
        df_seperators = self.seperator(image,df_result)
        return df_result, df_seperators


class VideoPlayerSeperator(ImagePlayerSeperator):
    def __init__(self,pretrained_model,url_to_model,display_result_to_gui):
        super(ImagePlayerSeperator,self).__init__( pretrained_model=pretrained_model,url_to_model=url_to_model,display_result_to_gui=display_result_to_gui )

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
                if self.display_result_to_gui:
                    cv2.imshow('Base Frame',frame)

                # Detect positions of human in image if any exist
                df_result, df_seperators = self.image_inference(frame)
                # print(f"df_result, df_seperators, {df_result}, {df_seperators}")
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