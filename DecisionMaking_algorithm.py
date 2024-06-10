from ultralytics import YOLO
import cv2
import numpy as np  
import pyrealsense2 as rs

def Silo_Detection():

    model = YOLO(r"models\best.engine")

    print(model.names)

    def point_inside(rectangle, entire_coordinates):    
        x1, y1 ,x2, y2 , _ , _ = rectangle
        return (entire_coordinates[0] <= x1 <= entire_coordinates[2] and entire_coordinates[1] <= y1 <= entire_coordinates[3]) or (entire_coordinates[0] <= x2 <= entire_coordinates[2] and entire_coordinates[1] <= y2 <= entire_coordinates[3])



    def decision_maker(image_input_def):
        global model
        model = YOLO(r"models\best.engine")

        balls_pattern = {'rack_1': {'blue_ball':0,'red_ball':0}, 
                        'rack_2': {'blue_ball':0,'red_ball':0}, 
                        'rack_3': {'blue_ball':0,'red_ball':0}, 
                        'rack_4': {'blue_ball':0,'red_ball':0}, 
                        'rack_5': {'blue_ball':0,'red_ball':0}}
        
        input_image = image_input_def
        results = model.predict(input_image,conf = 0.3,classes = [0,1,2])

        balls_detected_condinates = list(filter( lambda x: x[-1] != 2.0, results[0].boxes.numpy().data.tolist()))
        detected_silo_condinates = sorted(list(filter( lambda x: x[-1] == 2.0, results[0].boxes.numpy().data.tolist())),key=lambda x: x[0])

        if len(detected_silo_condinates) == 5: 
            
            finded_balls_in_silo = {}
            for id,i in enumerate(detected_silo_condinates):
                l = []
                d = {}
                for j in balls_detected_condinates:
                    res_inside = point_inside(j,i)
                    if res_inside:
                        l.append((j[-1]))
                blue_ball_count = l.count(0.0)
                red_ball_count = l.count(1.0)
                d['blue_ball'] = blue_ball_count
                d['red_ball'] = red_ball_count
                
                finded_balls_in_silo["rack_" + str(id + 1)] = d
                
            
            balls_pattern['rack_1'] = finded_balls_in_silo['rack_1']
            balls_pattern['rack_2'] = finded_balls_in_silo['rack_2']
            balls_pattern['rack_3'] = finded_balls_in_silo['rack_3']
            balls_pattern['rack_4'] = finded_balls_in_silo['rack_4']
            balls_pattern['rack_5'] = finded_balls_in_silo['rack_5']



            if 1 == 2:
                o = 1
                
            emtpy_racks = []
            my_ball_check_count = 0

            # finding empty racks
            for rack,ball_count in balls_pattern.items():
                if ball_count['blue_ball'] != 0:
                    my_ball_check_count += 1
                if ball_count['red_ball'] == 0 and ball_count['red_ball'] == 0:
                    emtpy_racks.append(rack)

            # removing filled in racks
            for rack,ball_count in list(balls_pattern.items()):
                total_ball_count_in_rack = ball_count['blue_ball'] + ball_count['red_ball']
                if total_ball_count_in_rack == 3:
                    balls_pattern.pop(rack)

            print(balls_pattern)
            max_ball_list = []
            # rack finding logic
            if len(emtpy_racks) == 0 or my_ball_check_count >= 3:
                
                # print("FIRST LOOP entered in after 3 ball placed in rack !!!!!!!!!!!!!!!!")
                
                for rack,ball_count in balls_pattern.items():
                    count_my_ball = ball_count['blue_ball']
                    count_opponent_ball = ball_count['red_ball']
                    
                    if count_my_ball == 1 and count_opponent_ball == 1:
                        sub_result = 2
                    else:
                        sub_result = count_my_ball - count_opponent_ball
                    # print(balls_pattern[rack], "m",count_my_ball,'o',count_opponent_ball)
                    max_ball_list.append((rack,sub_result))
                    
                
            if my_ball_check_count < 3 :
                
                # print("SECOND LOOP entered in before first 3 ball placed in rack !!!!!!!!!!!!!!!!")
                
                for rack,ball_count in balls_pattern.items():
                    count_my_ball = ball_count['red_ball']
                    count_opponent_ball = ball_count['blue_ball']
                    
                    if count_my_ball == 1 and count_opponent_ball == 1:
                        sub_result = 2
                        max_ball_list.append((rack,sub_result))
                    
            if max_ball_list == []:
                keys_position = emtpy_racks[0]
            else:
                keys_position = max(max_ball_list, key=lambda x: x[1])[0]
                
        else:
            keys_position = 0        
            
        final_image_result  = results[0].plot()
        final_image_result = cv2.putText(final_image_result,str(keys_position),(450,400),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=1)
        return keys_position,final_image_result


    W = 640
    H = 480

    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # model = YOLO(r"models\best.engine")

    min_z_coordinate_blue = None
    min_z_coordinate_purple = None
    color_to_detect = 'b'  # Start by detecting blue

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        result_silo,result_frame = decision_maker(color_image)


        cv2.imshow("Result",result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(result_silo)

    pipeline.stop()
    cv2.destroyAllWindows()
if  __name__ == '__main__':
    Silo_Detection()