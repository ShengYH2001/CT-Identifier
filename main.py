import os
import time
import math
import statistics
import multiprocessing
from PIL import Image
import numpy as np



def main(img_path: str):

    img: Image.Image = Image.open(img_path)
    img_array: np.ndarray = np.array(img)
    
    if img_array.shape[2] == 3:
        img_list = list(img_array)
        for i in range(len(img_list)):
            img_list[i] = list(img_list[i])
            for j in range(len(img_list[i])):
                img_list[i][j] = np.array([img_list[i][j][k] for k in range(3)] + [255])
        img_array = np.array(img_list)
    

    # 比例尺RGB=[255,128,128,255]
    is_scale_array: np.ndarray = np.zeros((len(img_array[:,0]),len(img_array[0,:])))
    is_scale_start = 0
    is_scale_continue = 5
    thr = 20
    thrr = 1.1
    for i in range(len(img_array[:,0])):
        for j in range(len(img_array[0,:])):
            # 若直接使用np.ndarray的"=="方法需加用"all()"方法转换为bool
            p = img_array[i,j].tolist()
            # is_scale_bool: bool = img_array[i,j].tolist() != [0,0,0,255] and img_array[i,j].tolist() != [0,0,0,0]; img_array[i,j]: np.ndarray
            is_scale_bool: bool = p[0] > thr and p[1] > thr and p[2] > thr
            is_scale_array[i,j] = is_scale_bool
            # 标记比例尺，以人工审核识别结果
            if is_scale_bool:
                img_array[i,j] = np.array([64,64,64,255])
                is_scale_start = 1
        if is_scale_start:
            if sum([p[0] < thr and p[1] < thr and p[2] < thr for p in img_array[i,:].tolist()]) / len(img_array[i,:]) > 0.97:
                is_scale_continue -= 1
                if not is_scale_continue:
                    break
                        
    # 比例尺为杠铃形，故识别底端
    is_scale = np.where(is_scale_array == True)
    # is_scale_end = is_scale[1][np.where(is_scale[0]==max(is_scale[0]))]
    scale_row = statistics.mode(is_scale[0])
    # scale_all_width = max(is_scale[1])-min(is_scale[1])
    scale_line_width = np.max(np.where(is_scale_array[scale_row,:])) - np.min(np.where(is_scale_array[scale_row,:]))
    scale_dot_height = 0 # 火柴头的厚度
    for i in range(min(is_scale[1]),max(is_scale[1])):
        height = np.max(np.where(is_scale_array[:,i])) - np.min(np.where(is_scale_array[:,i]))
        if height >= scale_dot_height:
            scale_dot_height = height
        else:
            break
    scale_len = scale_line_width - scale_dot_height
    # 标记识别的比例尺长度
    img_array[max(is_scale[0]), np.min(np.where(is_scale_array[scale_row,:]))+math.ceil(scale_dot_height/2):np.max(np.where(is_scale_array[scale_row,:]))-math.floor(scale_dot_height/2)] = np.array([255,255,255,255])
    img_array[max(is_scale[0])+1, np.min(np.where(is_scale_array[scale_row,:]))+math.ceil(scale_dot_height/2):np.min(np.where(is_scale_array[scale_row,:]))+math.ceil(scale_dot_height/2)+scale_len] = np.array([210,255,210,255])

    # 背景色RGB=[x,x,x,255]，前景色根据RGB比例识别
    red_num = 0
    green_num = 0
    blue_num = 0
    for i in range(np.max(is_scale[0])+1, len(img_array[:,0])):
        for j in range(len(img_array[0,:])):
            p = img_array[i,j].tolist()
            total_ratio: float = img_array[i,j][0] + img_array[i,j][1] + img_array[i,j][2]
            red_ratio: float = img_array[i,j][0]*3/total_ratio
            green_ratio: float = img_array[i,j][1]*3/total_ratio
            blue_ratio: float = img_array[i,j][2]*3/total_ratio
            if p[0] < thr and p[1] < thr and p[2] < thr:
                continue
            if red_ratio > thrr and red_ratio > green_ratio and red_ratio > blue_ratio:
                img_array[i,j] = np.array([255,0,0,255])
                red_num += 1
            elif green_ratio > thrr and green_ratio > red_ratio and green_ratio > blue_ratio:
                img_array[i,j] = np.array([0,255,0,255])
                green_num += 1
            elif blue_ratio > thrr and blue_ratio > green_ratio and blue_ratio > red_ratio:
                img_array[i,j] = np.array([0,0,255,255])
                blue_num += 1

    img_output: Image.Image = Image.fromarray(np.uint8(img_array))
    img_output.save(".\\result\\" + img_path.split(".png")[0].rsplit("\\",maxsplit=1)[-1] + "_result" + ".png")

    result: list[str,float] = [img_path.split(".png")[0].rsplit("\\",maxsplit=1)[-1], red_num/(scale_len**2), green_num/(scale_len**2), blue_num/(scale_len**2)]
    with open(".\\result.txt", "a") as file:
        file.write("\t".join(str(res) for res in result)+"\n")
    return result



if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    img_list = [".\\resource\\" + img for img in os.listdir(".\\resource") if img.endswith('.png')]
    
    # for i in img_list:
    #     main(i)
    
    start_time = time.time()

    pool = multiprocessing.Pool()
    pool.map(main, img_list)
    pool.close()
    pool.join()

    end_time = time.time()
    print("Time: ", end_time - start_time)
