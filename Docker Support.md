# Docker Support

**Author:** Yucheng Liang

The Dockerfile in this folder will build Docker images with all the dependencies that are necessary to run example notebooks.

---

## Building and Running with Docker

With Dockerfile, it's very easy to build a docker image for this project. At this project folder, run the following command:

```bash
docker build -t beta-recsys .
```

After running this command, a new image of this project will be built.

---

## Notes

1. If you are in China(Mainland), you may encounter issues about network. Downloading anaconda is relatively slow, so you can use mirror image in Line18 to substitute the url in Line 19 as follows:

   ```dockerfile
   # Before
   # Anaconda installing
   # Mirror: https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.0.1-Linux-x86_64.sh
   RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.0.1-Linux-x86_64.sh
   RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
   RUN rm Anaconda3-5.0.1-Linux-x86_64.sh
   ```

