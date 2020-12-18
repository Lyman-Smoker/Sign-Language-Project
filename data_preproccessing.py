from sklearn.externals import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
#读入文件
js=pd.read_json(path_or_buf='../../1413446_openpose.json')
#获取其中的frames
frames=js['frames']
#该json文件中只有b摄像机和c摄像机拍到的数据，因此我们获取b摄像机拍到的数据
frames=frames[0]
#定义
train_data=np.array([])
train_confidence=np.array([])
train_is_signing=np.zeros(len(frames),dtype=int)
#如果该摄像头拍摄的人在第4帧到第6帧在做手语，则将其标记为1
train_is_signing[4:7]=1
#对每一帧进行处理
for i in range(len(frames)):
  frame=frames[str(i)]
  people=frame['people'][0]
    # print(people)
  pose_keypoints_2d = np.array(people['pose_keypoints_2d'])
  face_keypoints_2d = np.array(people['face_keypoints_2d'])
  hand_left_keypoints_2d = np.array(people['hand_left_keypoints_2d'])
  hand_right_keypoints_2d = np.array(people['hand_right_keypoints_2d'])
  pose_keypoints_2d = pose_keypoints_2d.reshape((25, 3))
  face_keypoints_2d = face_keypoints_2d.reshape((70, 3))
  hand_left_keypoints_2d = hand_left_keypoints_2d.reshape((21, 3))
  hand_right_keypoints_2d = hand_right_keypoints_2d.reshape((21, 3))
  data = np.concatenate([pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d])
  x, confidence = np.hsplit(data, [2])
  x=np.array([x],dtype=np.float32)
  confidence=np.array([confidence],dtype=np.float32)
  confidence=confidence.T
  if(i==0):
    train_data=np.array([x])
    train_confidence=np.array(confidence)
  else:
    train_data=np.concatenate([train_data,np.array([x])])
    train_confidence=np.concatenate([train_confidence,np.array(confidence)])
    print(i)

#将获取的数据存入字典（注意方法为： 摄像机编号_of_文件名）
train_is_signing.astype('byte')
b1_of_1413446_openpose={}
b1_of_1413446_openpose['train_data']=train_data
b1_of_1413446_openpose['train_confidence']=train_confidence
b1_of_1413446_openpose['train_is_signing']=train_is_signing
#存储数据到m文件
joblib.dump(b1_of_1413446_openpose,'b1_of_1413446_openpose_for_train.m')
# d=joblib.load('b1_of_1413446_openpose_for_train.m')
