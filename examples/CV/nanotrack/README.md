# NanoTrack

## 1.模型获取



```shell
cd model
sh download_model.sh
```



## 2.Demo

输入输出数据说明:

```
输入:执行时--video_name指向视频地址或者默认使用摄像头;main.cc中init_rect的追踪目标的坐标值(左上角，右下角)
输出:追踪目标的新坐标值(左上角，右下角)
```



### 2.1 c++ Demo

```shell
cd cpp
mkdir build
cd build 
cmake ..
make -j8
./nanotrack_demo --video_name ../../data/1.mp4
```

