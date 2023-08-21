import os
import shutil
import json
import numpy as np
import torch
import time

source_directory = '/content/drive/MyDrive/pcaliber/label'  # Replace with the actual source directory path
target_directory = '/content/drive/MyDrive/pcaliber/label/json'      # Replace with the actual target directory path

# Start measuring time
start_time = time.time()


ans = [] #모든 annotation data 들어가는 리스트
ans_y = [] #모든 action data 들어가는 리스트
action_list = ['BODYLOWER','BODYSCRATCH','BODYSHAKE','FEETUP','FOOTUP','HEADING','LYING','MOUNTING','SIT','TAILING','TAILLOW','WALKRUN','TURN']
for action in action_list:
  folder_path = f'/content/drive/MyDrive/pcaliber/label/{action}/'
  for file in os.listdir(folder_path):
    if file.endswith(".json"):
      file_path = os.path.join(folder_path,file)
      with open(file_path,"r") as json_file:
        data = json.load(json_file)
        #모든 json file의 annotations data 60 frame 까지만 넣기
        #한 json file의 어노테이션 값 넣을 리스트 생성, 한 파일당 60개의 리스트 생성됨
        if "annotations" in data:
              annotations = data["annotations"]
              if len(annotations)>=60:
                all_frame_ann = []
                for m in range(60):
                  now = annotations[m]['keypoints']
                  class_ann = []
                  for i in range(1,16):
                    xy_or_none = now[str(i)]
                    if xy_or_none != None:
                      class_ann.append([xy_or_none["x"],xy_or_none["y"]])
                      # ann.append(x_y)
                    else:
                      class_ann.append([0,0])
                  all_frame_ann.append(class_ann)
                ans.append(all_frame_ann)
                # y값 넣기
                if "metadata" in data:
                    k = data["metadata"]
                    ans_y.append(action)
                else:
                  continue
print(len(ans_y))
print(len(ans))
print(ans[0])

# End measuring time
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")