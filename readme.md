# Cinematographic Style Analysis

Here is the code for film style analysis used in my graduate thesis. I use main color, light tone, saturation, light position, shot scale, symmetry, shot duration and camera motion features to categorize directors. Detail of those features mentioned above can be found in `/methods/`.

## Structure&Usage

The codes are divided into 2 parts: feature extraction and feature analysis. Codes for feature extraction are mainly in `/methods/` while codes for feature analysis are in `/synthesis/`.

### Feature Extraction

The main entrance of feature extraction is `./AnalyseVideo.py`.

Try the following code to extract features from a single file or multiple videos.

```shell
python ./AnalyseVideo.py --video_path {videofolder or file path}
```

The feature output will be saved under the same directory to the video file, with the name `{$basename}.output`. More options can be adjusted in `./methods/opt.py`. I do not recommend large batch size and small sample intervals because several neural networks are used when running.

The output file contains multiple static frame features of a single video file. Each block, which represents a single frame, contains 12 lines with the following structure:

```
main color1  [%3d %3d %3d]
main color2  [%3d %3d %3d]
main color3  [%3d %3d %3d]
main color4  [%3d %3d %3d]
main color5  [%3d %3d %3d]
light tone   [%d]
saturation   [%d]
symmetry     [%.2f %.2f]
human num    [%d]
shot scale   [%d]
lightpos idx [%d]
lightpos sc  [%.2f]
```

Codes in `./utils/` are for data preparation and training process. They will not be used in the feature extracting process (which is basically inference). Codes in `./test/` are for debug and single feature output. 

Camera motion descriptor is not based on frames but on video clips. Use `make` command to build `homography.cpp` and then run `extractCameraDescriptor.sh`. You should assign a video folder instead of single video file. Read it for more details.

### Feature Analysis

In my analysis, only shot scale, shot duration, light position, saturation, tone, main color are used in the final visualization and classification. 

To begin with, you need to put index file in `./synthesis/indexfile/` and outputs folders in `./synthesis/outputs/`. Remember the corresponding files should has the same name regarding CAP.

You can run `shot_lenght.m` to compute and visualize a single movie (index file).

Run `read_cam.m` to read and visualize camera motion feature.

For other features, you should run `read_multiple.m` (and adjust the input and output name manually) to compute summarized features. While then run `process_all.m` to compute final features for the entire input movies. Then, run `vis.m` or `classify.py` to visualize (using t-SNE) or classify (using SVM). You could also run `process_*.m` to see the features of each input movies.

## Future works

- Try Semantic Line Detection methods for structure features.
- Try to classify camera motion features into higher and more meaningful (controllable) features.
- Genre influence.
- Develop shot scale detection method with wider applicability. 
- More accurate labels.
- Try Element Detection (snow, fire, etc.)