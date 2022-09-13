import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

# input
parser = argparse.ArgumentParser()
parser.add_argument('--hand', required=True)
parser.add_argument('--session', required=True)
parser.add_argument('--study_frame', default=500)
parser.add_argument('--pause_time', default=5)
args = parser.parse_args()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

os.makedirs('Datasets/', exist_ok=True)

gesture_list = ['NONE', 'ULF', 'UMF', 'URF', 'DLF', 'DMF', 'DRF', 'ULO', 'URO', 'DLO', 'DRO', 'ULB', 'UMB', 'URB', 'DLB', 'DMB', 'DRB', 'TON']

hand = args.hand
session = int(args.session)
study_frame = int(args.study_frame)
pause_time = int(args.pause_time)


print('::Brushing teeth  Procedure::')
print(hand, end='\t')
for gesture in gesture_list:
    if gesture==gesture_list[-1]:
        print(gesture, end='')
    else:
        print(gesture, end='')
        print('->', end='')
print()

for gesture in gesture_list:
    cap = cv2.VideoCapture(0)

    position = np.zeros((1,21,3))
    velocity = np.zeros((1,21,3))
    acceleration = np.zeros((1,21,3))

    with mp_hands.Hands(model_complexity=1,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:



        print('::{} {} Ready.... '.format(hand,gesture))
        cnt=pause_time

        for i in [1000]*pause_time:
            print(cnt)
            cv2.waitKey(i)
            cnt-=1

        print('Start')

        while len(position) <= study_frame:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

            frame_vector_x = []
            frame_vector_y = []
            frame_vector_z = []
            if results.multi_hand_landmarks:
                for point in results.multi_hand_landmarks[0].landmark:
                    frame_vector_x.append(point.x)
                    frame_vector_y.append(point.y)
                    frame_vector_z.append(point.z)
                frame_array = np.vstack((frame_vector_x,frame_vector_y,frame_vector_z)).reshape(1,21,3)

                position = np.vstack((position,frame_array))

                if len(position)>2:
                    current_velocity = position[-1] - position[-2]
                    velocity=np.vstack((velocity, current_velocity.reshape(1,21,3)))

                if len(position)>3:
                    current_acceleration = velocity[-1] - velocity[-2]
                    acceleration=np.vstack((acceleration, current_acceleration.reshape(1,21,3)))


            cv2.putText(frame,'Dataset Count : %d' % (position.shape[0]-1),(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        position=position[1:]
        velocity=velocity[1:]
        acceleration=acceleration[1:]


        np.save('Datasets/%s_%s_p_%d' % (hand,gesture, session),position)
        np.save('Datasets/%s_%s_v_%d' % (hand,gesture, session),velocity)
        np.save('Datasets/%s_%s_a_%d' % (hand,gesture, session),acceleration)
        cv2.waitKey(1000)


    cap.release()
    cv2.destroyAllWindows()
    print('Finished')