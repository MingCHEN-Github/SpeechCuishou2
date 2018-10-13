import os, wave, audioop
#若in和out有立体声
def downsampleWavStero(src, dst, inrate=44100, outrate=8000, inchannels=2, outchannels=1):

    if not os.path.exists(src):
        print('Source not found!')
        return False
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print('Failed to open files!')
        return False
    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)
    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print('Failed to downsample wav')
        return False
    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted)
    except:
        print('Failed to write wav')
        return False
    try:
        s_read.close()
        s_write.close()
    except:
        print('Failed to close wav files')
        return False
    return True
#若in和out都是单通道
def downsampleWavMono(src, dst, inrate=48000, outrate=16000, inchannels=1, outchannels=1):
    import os, wave, audioop
    if not os.path.exists(src):
        print('Source not found!')
        return False
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    try:
        s_read = wave.open(src, 'rb')
        params = s_read.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        print(nchannels, sampwidth, framerate, nframes)
        s_write = wave.open(dst, 'wb')
    except:
        print('Failed to open files!')
        return False
    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1 and inchannels != 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print('Failed to downsample wav')
        return False
    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted[0])
    except Exception as e:
        print(e)
        print('Failed to write wav')
        return False
    try:
        s_read.close()
        s_write.close()
    except:
        print('Failed to close wav files')
        return False
    return True


#计算，转换44.1khz到8khz
def src2dst_AudioFiles(filepath_src, filepath_dst):
    audioPath_src=[]
    #audioPath_dst=[]
    filename1_src = os.listdir(filepath_src)
   # filename1_dst = os.listdir(filepath_dst)
    for file1_src in filename1_src:
        # 新建目标文件夹中不存在的文件
        if not os.path.exists(filepath_dst + file1_src):
            os.mkdir(filepath_dst + file1_src)
        filename2_src = os.listdir(filepath_src+file1_src)
        for file2_src in filename2_src:
            # 新建目标文件夹中不存在的文件
            if not os.path.exists(filepath_dst + file1_src+"\\"+file2_src):
                os.mkdir(filepath_dst + file1_src+"\\"+file2_src)
            filename3_src = os.listdir(filepath_src+file1_src+"\\"+file2_src)
            for file3_src in filename3_src:
                audioPath_src.append(filepath_src+file1_src+"\\"+file2_src+"\\"+file3_src)
                #print(filepath+file1+"\\"+file2+"\\"+file3)
                eachWavPath_src=filepath_src+file1_src+"\\"+file2_src+"\\"+file3_src #当前wav的绝对路径
                eachWavPath_dst =filepath_dst+file1_src+"\\"+file2_src+"\\"+file3_src
                print(eachWavPath_src)

                #转换44.1khz到8khz
                downsampleWavStero(eachWavPath_src,eachWavPath_dst,
                                   inrate=44100, outrate=8000, inchannels=2, outchannels=1) #转换当前wav


#调用函数进行计算
Path_src = 'C:\\Users\\zkycs3\\Desktop\\audio8k_src\\'

Path_dst = 'C:\\Users\\zkycs3\\Desktop\\audio8k_dst\\'
src2dst_AudioFiles(Path_src,Path_dst)
