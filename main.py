import torch #import torch library
from inference import * 
# Model
if __name__=="__main__":    
    input_file = "./input/video/deepsort_30sec.mp4"
    video_player_seperator = VideoPlayerSeperator(pretrained_model=False,url_to_model='./model/weight/yolov5s_testing_4.pt')
    video_player_seperator.video_inference(input_file)
    # print(video_player_seperator.test_model_connection())